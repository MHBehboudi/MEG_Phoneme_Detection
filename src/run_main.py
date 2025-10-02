# src/run_main.py

import os
import argparse
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# Import modules
from config.default_config import *
from src.system.classifier import PhonemeClassifier, PrintValTwoF1
from src.data.data_wrappers import build_loaders, per_sample_z
from pnpl.datasets import LibriBrainCompetitionHoldout


# ----------------------------
# Small utils (needed in main)
# ----------------------------
def now_str():
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_shifts(s: str) -> List[int]:
    vals = []
    for t in s.split(','):
        t = t.strip()
        if not t:
            continue
        vals.append(int(t))
    if 0 not in vals:
        vals.append(0)
    return sorted(list(set(vals)))

# ============================================================
# ==================== TRAIN / EVAL LOOP =====================
# ============================================================
def run_stage(phase_name: str, loaders: dict, num_epochs: int, learning_rate: float,
              logger_version: str) -> str:
    model = PhonemeClassifier(
        class_counts=loaders["class_counts"],
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=BATCH_SIZE,
        per_sample_zscore_train=False, # Z-score flag passed to classifier constructor
        per_sample_zscore_val_avg=loaders["val_flags"]["z_avg"],
        per_sample_zscore_val_single=loaders["val_flags"]["z_single"],
    )
    logger = CSVLogger(save_dir=f"{BASE_PATH}/lightning_logs", name=logger_version, version="")
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{BASE_PATH}/models",
        filename=f"best_lstm_v12_{phase_name}" + "_{epoch:02d}-{val_f1_macroK100:.3f}",
        monitor="val_f1_macroK100", mode="max", save_top_k=1, save_last=True
    )
    early = EarlyStopping(monitor="val_f1_macroK100", mode="max", patience=10, verbose=True)
    trainer = L.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=MIXED_PRECISION,
        max_epochs=num_epochs,
        logger=logger,
        callbacks=[ckpt_cb, TQDMProgressBar(refresh_rate=1), PrintValTwoF1(), early],
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )
    print(f"[{now_str()}] === {logger_version}: epochs={num_epochs}, LR={learning_rate} ===")
    print(f"[{now_str()}] >>> Fitting {logger_version} ...")
    trainer.fit(model, loaders["train"], [loaders["val_avg"], loaders["val_single"]])
    best = trainer.checkpoint_callback.best_model_path or trainer.checkpoint_callback.last_model_path
    print(f"[{now_str()}] --- {logger_version} best model at: {best} ---")
    return best

def build_classifier(ckpt_path: str, val_flags: dict) -> PhonemeClassifier:
    dummy_counts = np.ones(NUM_PHONEME_CLASSES, dtype=np.int64)  # not used at infer
    model = PhonemeClassifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        class_counts=dummy_counts,
        learning_rate=1e-6,
        num_epochs=1,
        batch_size=BATCH_SIZE,
        per_sample_zscore_train=False,
        per_sample_zscore_val_avg=val_flags.get("z_avg", False),
        per_sample_zscore_val_single=val_flags.get("z_single", False),
    )
    return model.eval()

@torch.no_grad()
def dump_probs(model: PhonemeClassifier, device: str,
               val_avg_loader, val_single_loader, tta_shifts: List[int],
               per_sample_zscore_holdout: bool, batch_size: int):
    os.makedirs(PRED_PATH, exist_ok=True)
    # ... (Omitted full dump_probs logic for brevity, copy the rest of this function from your original code)
    
    # 1) Validation K=100
    probs_val_avg = []
    for x, _gnn, y in val_avg_loader:
        x = x.to(device).float()
        x = per_sample_z(x) if model.z_val_avg else x
        logits = model.model(x)
        probs = F.softmax(logits, dim=-1)
        probs_val_avg.append(probs.cpu().numpy())
    probs_val_avg = np.concatenate(probs_val_avg, axis=0)
    np.save(f"{PRED_PATH}/lstm_v12_val_avg_probs.npy", probs_val_avg)

    # 2) Validation K=1
    probs_val_single = []
    for x, _gnn, y in val_single_loader:
        x = x.to(device).float()
        x = per_sample_z(x) if model.z_val_single else x
        logits = model.model(x)
        probs = F.softmax(logits, dim=-1)
        probs_val_single.append(probs.cpu().numpy())
    probs_val_single = np.concatenate(probs_val_single, axis=0)
    np.save(f"{PRED_PATH}/lstm_v12_val_single_probs.npy", probs_val_single)

    # 3) Holdout with optional per-sample z-score + TTA
    holdout_ds = LibriBrainCompetitionHoldout(data_path=f"{BASE_PATH}/data/", task="phoneme")
    class HoldWrap(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i): return self.base[i], torch.empty(0)
    loader = DataLoader(HoldWrap(holdout_ds), batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    probs_hold = []
    for x, _ in loader:
        x = x.to(device).float()
        if per_sample_zscore_holdout:
            x = per_sample_z(x)
        acc = None
        for s in tta_shifts:
            xs = torch.roll(x, shifts=s, dims=-1) if s != 0 else x
            logits = model.model(xs)
            p = F.softmax(logits, dim=-1)
            acc = p if acc is None else (acc + p)
        p_mean = acc / float(len(tta_shifts))
        probs_hold.append(p_mean.cpu().numpy())
    probs_hold = np.concatenate(probs_hold, axis=0)
    np.save(f"{PRED_PATH}/lstm_v12_holdout_probs.npy", probs_hold)


# ============================================================
# ============================ MAIN ==========================
# ============================================================
def main():
    parser = argparse.ArgumentParser("LibriBrain Phoneme — LSTM_v12 train + submit")
    parser.add_argument("--mode", type=str, default="train_and_submit", choices=["train", "submit", "train_and_submit"])
    parser.add_argument("--stage1_epochs", type=int, default=8)
    parser.add_argument("--stage2_epochs", type=int, default=6)
    parser.add_argument("--stage2_lr", type=float, default=LEARNING_RATE_STAGE2)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--precision", type=str, default=MIXED_PRECISION)
    parser.add_argument("--multik_stage2", action="store_true", help="Use Multi-K in Stage-2 instead of fixed K=100")
    # normalization/tta
    parser.add_argument("--per_sample_zscore_train", action="store_true")
    parser.add_argument("--per_sample_zscore_val_avg", action="store_true")
    parser.add_argument("--per_sample_zscore_val_single", action="store_true")
    parser.add_argument("--per_sample_zscore_holdout", action="store_true")
    parser.add_argument("--tta_shifts", type=str, default="0,2,-2,4,-4")
    args = parser.parse_args()

    L.seed_everything(42)
    for path in [f"{BASE_PATH}/data/", f"{BASE_PATH}/lightning_logs/", f"{BASE_PATH}/models/", SUBMISSION_PATH, PRED_PATH]:
        os.makedirs(path, exist_ok=True)

    print(f"Job started at {now_str()}")
    print(f"Python: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

    tta_shifts = parse_shifts(args.tta_shifts)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- TRAIN ----------
    ckpt_stage2 = None
    if args.mode in ["train", "train_and_submit"]:
        # Stage-1: K=32
        print(f"\n[{now_str()}] --- Phase 1: Train K={TRAIN_AVG_K_STAGE1}, Val K={VAL_AVG_K} + single (K=1) ---")
        loaders_p1 = build_loaders(
            base_path=BASE_PATH, batch_size=args.batch_size,
            train_mode="fixed", k_value=TRAIN_AVG_K_STAGE1,
            val_k=VAL_AVG_K, drop_remaining=DROP_REMAINING, standardize_wrapper=WRAPPER_STANDARDIZE,
            per_sample_zscore_val_avg=args.per_sample_zscore_val_avg,
            per_sample_zscore_val_single=args.per_sample_zscore_val_single
        )
        print(f"[{now_str()}] Built loaders (Stage-1). Starting training...")
        best_p1 = run_stage(
            phase_name="stage1_k32", loaders=loaders_p1,
            num_epochs=args.stage1_epochs, learning_rate=LEARNING_RATE_STAGE1,
            logger_version=f"phoneme_stage1_k32_lstm_v12"
        )

        # Stage-2: finetune
        print(f"\n[{now_str()}] --- Phase 2: {'Multi-K {60..100}' if args.multik_stage2 else 'Fixed K=100'} (Val K=100 + single) ---")
        loaders_p2 = build_loaders(
            base_path=BASE_PATH, batch_size=args.batch_size,
            train_mode=("multik" if args.multik_stage2 else "fixed"),
            k_value=TRAIN_AVG_K_STAGE2_FIXED,
            multik_min=MULTIK_MIN, multik_max=MULTIK_MAX, multik_step=MULTIK_STEP,
            val_k=VAL_AVG_K, drop_remaining=DROP_REMAINING, standardize_wrapper=WRAPPER_STANDARDIZE,
            per_sample_zscore_val_avg=args.per_sample_zscore_val_avg,
            per_sample_zscore_val_single=args.per_sample_zscore_val_single
        )
        print(f"[{now_str()}] Built loaders (Stage-2). Loading Stage-1 ckpt & fine-tuning...")

        model = PhonemeClassifier.load_from_checkpoint(
            checkpoint_path=best_p1,
            class_counts=loaders_p2["class_counts"],
            learning_rate=args.stage2_lr,
            num_epochs=args.stage2_epochs,
            batch_size=args.batch_size,
            per_sample_zscore_train=args.per_sample_zscore_train,
            per_sample_zscore_val_avg=loaders_p2["val_flags"]["z_avg"],
            per_sample_zscore_val_single=loaders_p2["val_flags"]["z_single"],
        )
        logger = CSVLogger(save_dir=f"{BASE_PATH}/lightning_logs", name="phoneme_stage2_finetune_lstm_v12", version="")
        ckpt_cb = ModelCheckpoint(
            dirpath=f"{BASE_PATH}/models",
            filename=f"best_lstm_v12_stage2" + "_{epoch:02d}-{val_f1_macroK100:.3f}",
            monitor="val_f1_macroK100", mode="max", save_top_k=1, save_last=True
        )
        early = EarlyStopping(monitor="val_f1_macroK100", mode="max", patience=10, verbose=True)
        trainer = L.Trainer(
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            precision=args.precision,
            max_epochs=args.stage2_epochs,
            logger=logger,
            callbacks=[ckpt_cb, TQDMProgressBar(refresh_rate=1), PrintValTwoF1(), early],
            check_val_every_n_epoch=1,
            gradient_clip_val=1.0,
            log_every_n_steps=50,
        )
        print(f"[{now_str()}] >>> Fitting Stage-2 (lstm_v12) ...")
        trainer.fit(model, loaders_p2["train"], [loaders_p2["val_avg"], loaders_p2["val_single"]])
        ckpt_stage2 = trainer.checkpoint_callback.best_model_path or trainer.checkpoint_callback.last_model_path
        print(f"[{now_str()}] --- Stage-2 best model at: {ckpt_stage2} ---")

    # If submit only or after training, try to find ckpt if not set
    if ckpt_stage2 is None:
        mdir = f"{BASE_PATH}/models"
        cand = []
        if os.path.isdir(mdir):
            for f in os.listdir(mdir):
                if f.startswith("best_lstm_v12_stage2") and f.endswith(".ckpt"):
                    cand.append(os.path.join(mdir, f))
        if not cand:
            raise RuntimeError(f"No Stage-2 checkpoint found for lstm_v12. Train first or place ckpt in {BASE_PATH}/models/")
        cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        ckpt_stage2 = cand[0]

    # ---------- SUBMIT (and dump preds) ----------
    if args.mode in ["submit", "train_and_submit"]:
        print(f"\n[{now_str()}] --- Dumping predictions & Submit (single-model) ---")
        # Build validation loaders once (for dumping val probs)
        loaders_val = build_loaders(
            base_path=BASE_PATH, batch_size=args.batch_size,
            train_mode="fixed", k_value=TRAIN_AVG_K_STAGE1,  # not used; only care about val loaders
            val_k=VAL_AVG_K, drop_remaining=DROP_REMAINING, standardize_wrapper=WRAPPER_STANDARDIZE,
            per_sample_zscore_val_avg=args.per_sample_zscore_val_avg,
            per_sample_zscore_val_single=args.per_sample_zscore_val_single
        )
        model = build_classifier(ckpt_stage2, loaders_val["val_flags"]).to(device)
        dump_probs(
            model=model, device=device,
            val_avg_loader=loaders_val["val_avg"], val_single_loader=loaders_val["val_single"],
            tta_shifts=parse_shifts(args.tta_shifts),
            per_sample_zscore_holdout=args.per_sample_zscore_holdout,
            batch_size=args.batch_size
        )

        # Build submission CSV directly from single-model holdout probs
        os.makedirs(SUBMISSION_PATH, exist_ok=True)
        out_csv = f"{SUBMISSION_PATH}/libribrain_phoneme_submission.csv"
        holdout_ds = LibriBrainCompetitionHoldout(data_path=f"{BASE_PATH}/data/", task="phoneme")
        holdout_probs = np.load(f"{PRED_PATH}/lstm_v12_holdout_probs.npy")
        ens_list = [torch.from_numpy(holdout_probs[i]) for i in range(holdout_probs.shape[0])]
        holdout_ds.generate_submission_in_csv(ens_list, out_csv)
        print(f"[✓] Submission file written to {out_csv}")

    print(f"[{now_str()}] Done.")

if __name__ == "__main__":
    main()
