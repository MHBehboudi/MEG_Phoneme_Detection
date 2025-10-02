# src/system/classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import classes and constants
from src.modules.model_v12 import PhonemeModelV12
from src.data.data_wrappers import per_sample_z
from config.default_config import (
    NUM_PHONEME_CLASSES, CONV_DIM, ATTN_DIM, LSTM_LAYERS, BI_DIRECTIONAL, DROPOUT_RATE,
    BATCH_SIZE
)

# ============================================================
# ==================== LOSS & METRICS ========================
# ============================================================
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts, num_classes, beta=0.999, gamma=1.5, eps=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.eps = eps
        counts = torch.tensor(class_counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=1.0)
        effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        self.register_buffer("alpha", weights)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (probs * one_hot).sum(dim=1).clamp_(min=self.eps, max=1.0)
        alpha_t = self.alpha[targets]
        loss = -alpha_t * torch.pow(1.0 - p_t, self.gamma) * torch.log(p_t)
        return loss.mean()

# ============================================================
# ================ LIGHTNING: CALLBACKS ======================
# ============================================================
class PrintValTwoF1(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        k100 = trainer.callback_metrics.get("val_f1_macroK100", None)
        k1   = trainer.callback_metrics.get("val_single_f1_macroK1", None)
        if k100 is not None and k1 is not None:
            v1 = float(k100) if hasattr(k100, "__float__") else k100.item()
            v2 = float(k1)   if hasattr(k1, "__float__") else k1.item()
            print(f"[VAL] val_f1_macro(K=100)={v1:.4f} | val_single_f1_macro(K=1)={v2:.4f}")

# ============================================================
# ================ LIGHTNING: CLASSIFIER =====================
# ============================================================

class PhonemeClassifier(L.LightningModule):
    def __init__(self, class_counts, learning_rate, num_epochs, batch_size,
                 per_sample_zscore_train, per_sample_zscore_val_avg, per_sample_zscore_val_single):
        super().__init__()
        self.save_hyperparameters()

        self.model = PhonemeModelV12(conv_dim=CONV_DIM, attn_dim=ATTN_DIM,
                                     lstm_layers=LSTM_LAYERS, bi=BI_DIRECTIONAL, dropout=DROPOUT_RATE)

        self.loss_fn = ClassBalancedFocalLoss(class_counts=class_counts,
                                              num_classes=NUM_PHONEME_CLASSES,
                                              beta=0.999, gamma=1.5)
        # Metrics
        self.train_f1_macro = F1Score(task="multiclass", num_classes=NUM_PHONEME_CLASSES, average='macro')
        self.val_f1_macro_avg = F1Score(task="multiclass", num_classes=NUM_PHONEME_CLASSES, average='macro')   # dl 0
        self.val_acc_avg      = Accuracy(task="multiclass", num_classes=NUM_PHONEME_CLASSES)
        self.val_f1_macro_single = F1Score(task="multiclass", num_classes=NUM_PHONEME_CLASSES, average='macro') # dl 1
        self.val_acc_single      = Accuracy(task="multiclass", num_classes=NUM_PHONEME_CLASSES)

        # Flags
        self.z_train = per_sample_zscore_train
        self.z_val_avg = per_sample_zscore_val_avg
        self.z_val_single = per_sample_zscore_val_single

    def forward(self, raw_meg, _gnn_unused=None):
        return self.model(raw_meg)

    def _maybe_z(self, x, is_train, dataloader_idx=None):
        if is_train and self.z_train:
            return per_sample_z(x)
        if not is_train and dataloader_idx == 0 and self.z_val_avg:
            return per_sample_z(x)
        if not is_train and dataloader_idx == 1 and self.z_val_single:
            return per_sample_z(x)
        return x

    def _step(self, batch, stage, dataloader_idx=None):
        if stage in ("train", "val"):
            raw_meg, _gnn, y = batch
        else:
            raw_meg = batch
            y = None

        raw_meg = raw_meg.to(self.device).float()
        raw_meg = self._maybe_z(raw_meg, is_train=(stage=="train"), dataloader_idx=dataloader_idx)
        if y is not None:
            y = y.to(self.device)

        logits = self(raw_meg, None)
        if y is not None:
            loss = self.loss_fn(logits, y)
        else:
            loss = None

        if stage == "train":
            self.train_f1_macro.update(logits, y)
            self.log("train_f1_macro", self.train_f1_macro, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        elif stage == "val":
            if dataloader_idx == 0:
                self.val_f1_macro_avg.update(logits, y)
                self.val_acc_avg.update(logits, y)
            elif dataloader_idx == 1:
                self.val_f1_macro_single.update(logits, y)
                self.val_acc_single.update(logits, y)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", None)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, "val", dataloader_idx)

    def on_validation_epoch_end(self):
        k100 = self.val_f1_macro_avg.compute()
        k1   = self.val_f1_macro_single.compute()
        self.log("val_f1_macroK100", k100, prog_bar=True)
        self.log("val_single_f1_macroK1", k1, prog_bar=True)
        self.val_f1_macro_avg.reset(); self.val_acc_avg.reset()
        self.val_f1_macro_single.reset(); self.val_acc_single.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1)
        cosine = CosineAnnealingLR(optimizer, T_max=max(self.hparams.num_epochs - 1, 1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
