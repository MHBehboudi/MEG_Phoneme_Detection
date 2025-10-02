
#!/bin/bash -l

#SBATCH --job-name=phoneme_detection
#SBATCH --partition=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Job started on $(hostname) at $(date)"

module purge
module load miniconda

# Make conda usable in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /conda_env

PYTHON=conda_env/bin/python

echo "Python: $($PYTHON -c 'import sys; print(sys.executable)')"

# Ensure pnpl is up to date
$PYTHON -m pip install --upgrade "pnpl>=0.0.6"

# Run the new modular script
srun $PYTHON src/run_main.py \
  --mode train_and_submit \
  --stage1_epochs 15 \
  --stage2_epochs 30 --stage2_lr 2e-5 \
  --multik_stage2 \
  --batch_size 16 --precision 16-mixed \
  --per_sample_zscore_train \
  --per_sample_zscore_val_avg --per_sample_zscore_val_single \
  --per_sample_zscore_holdout \
  --tta_shifts "0,2,-2,4,-4,6,-6"

echo "Job finished at: $(date)"
