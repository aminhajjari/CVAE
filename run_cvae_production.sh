#!/bin/bash

#=======================================================================
# PRODUCTION SLURM - CVAE on All Tabular Datasets
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=CVAE_Production
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

#=======================================================================
# OUTPUTS + SLURM LOGS (ALL IN ONE PLACE)
#=======================================================================
#SBATCH --output=/home/gkianfar/scratch/Amin/Tab2Vis/outputs/job_logs/production_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Tab2Vis/outputs/job_logs/production_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration
#=======================================================================
BASE_DIR="/home/gkianfar/scratch/Amin/Tab2Vis"

DATASETS_DIR="$BASE_DIR/Unzippeddata/CSV"

# ✅ ALL RESULTS + ARTIFACTS HERE
OUTPUTS_DIR="$BASE_DIR/outputs"
RESULTS_BASE="$OUTPUTS_DIR/results"
JOB_LOGS_DIR="$OUTPUTS_DIR/job_logs"

# ✅ CHANGED: no more project/ subfolder
CVAE_DIR="$BASE_DIR/CVAE"

# ✅ CHANGED: venv is now directly under Tab2Vis/
VENV_PATH="$BASE_DIR/venvMsc/bin/activate"

BATCH_SCRIPT="$CVAE_DIR/run_all_datasets.py"
MAIN_SCRIPT="$CVAE_DIR/run_vif.py"

TIMEOUT_DEFAULT=7200

#=======================================================================
# Job Info
#=======================================================================
echo "=========================================="
echo "🧬 CVAE PRODUCTION RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

#=======================================================================
# Setup directories
#=======================================================================
mkdir -p "$JOB_LOGS_DIR"
mkdir -p "$RESULTS_BASE"

#=======================================================================
# Dataset check
#=======================================================================
if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ Dataset not found: $DATASETS_DIR"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"

#=======================================================================
# Modules + env
#=======================================================================
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

source "$VENV_PATH"

echo "Python version:"
python --version

#=======================================================================
# Run
#=======================================================================
echo "🚀 Starting training..."

python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT"

EXIT_CODE=$?

#=======================================================================
# Final
#=======================================================================
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS"
    echo "Results in: $RESULTS_BASE"
    echo "Logs in: $JOB_LOGS_DIR"
else
    echo "❌ FAILED - check logs:"
    echo "$JOB_LOGS_DIR"
fi

exit $EXIT_CODE
