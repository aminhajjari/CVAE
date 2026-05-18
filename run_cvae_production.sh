#!/bin/bash
#=======================================================================
# PRODUCTION SLURM - CVAE on ALL Tabular Datasets (Heavy Optimized)
#=======================================================================
#SBATCH --account=def-arashmoh
#SBATCH --job-name=CVAE_Production_All
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=72:00:00                    # 3 days - safe for full run
#=======================================================================
#SBATCH --output=/home/gkianfar/scratch/Amin/Tab2Vis/outputs/job_logs/production_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Tab2Vis/outputs/job_logs/production_%A.err
#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#=======================================================================

BASE_DIR="/home/gkianfar/scratch/Amin/Tab2Vis"
DATASETS_DIR="$BASE_DIR/Unzippeddata/CSV"
OUTPUTS_DIR="$BASE_DIR/outputs"
RESULTS_BASE="$OUTPUTS_DIR/results"
JOB_LOGS_DIR="$OUTPUTS_DIR/job_logs"

CVAE_DIR="$BASE_DIR/CVAE"
VENV_PATH="$BASE_DIR/venvMsc/bin/activate"
BATCH_SCRIPT="$CVAE_DIR/run_all_datasets.py"
MAIN_SCRIPT="$CVAE_DIR/run_vif.py"

# ====================== KEY CHANGES ======================
TIMEOUT_DEFAULT=21600                    # 6 hours per dataset (was 7200)
# ========================================================

echo "=========================================="
echo "🧬 CVAE PRODUCTION RUN - ALL DATASETS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Per-dataset timeout: ${TIMEOUT_DEFAULT}s (6 hours)"
echo "=========================================="

# Setup directories
mkdir -p "$JOB_LOGS_DIR"
mkdir -p "$RESULTS_BASE"

# Dataset check
if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ Dataset directory not found: $DATASETS_DIR"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"

# Modules + env
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
source "$VENV_PATH"

echo "Python version:"
python --version

#=======================================================================
# Run All Datasets
#=======================================================================
echo "🚀 Starting full training on all datasets..."

python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT" \
    --resume true                     # Add this if your script supports resume

EXIT_CODE=$?

#=======================================================================
# Final Status
#=======================================================================
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS - All datasets completed"
    echo "Results in: $RESULTS_BASE"
    echo "Logs in: $JOB_LOGS_DIR"
else
    echo "⚠️  Job finished with issues - check logs"
fi

exit $EXIT_CODE
