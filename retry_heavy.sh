#!/bin/bash
#=======================================================================
# RETRY - Only Bioresponse + CIFAR-10-tabular
#=======================================================================
#SBATCH --account=def-arashmoh
#SBATCH --job-name=CVAE_Retry_Heavy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00                    # 24h should be more than enough
#SBATCH --output=/home/gkianfar/scratch/Amin/Tab2Vis/outputs/job_logs/retry_heavy_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Tab2Vis/outputs/job_logs/retry_heavy_%A.err


BASE_DIR="/home/gkianfar/scratch/Amin/Tab2Vis"
DATASETS_DIR="$BASE_DIR/Unzippeddata/CSV"
OUTPUTS_DIR="$BASE_DIR/outputs"
VENV_PATH="$BASE_DIR/venvMsc/bin/activate"
BATCH_SCRIPT="$BASE_DIR/CVAE/run_all_datasets.py"
MAIN_SCRIPT="$BASE_DIR/CVAE/run_vif.py"

echo "=========================================="
echo "🧬 CVAE RETRY - HEAVY DATASETS ONLY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

mkdir -p "$OUTPUTS_DIR/job_logs"
mkdir -p "$OUTPUTS_DIR/results"

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
source "$VENV_PATH"

echo "Python version:"
python --version

echo "🚀 Retrying Bioresponse + CIFAR-10-tabular with higher timeout..."

python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$OUTPUTS_DIR/results" \
    --job_id "RETRY_$(date +%Y%m%d)" \
    --script_path "$MAIN_SCRIPT" \
    --timeout 28800 \                     # ← 8 hours per dataset (was 7200)
    --dataset-list "Bioresponse,CIFAR-10-tabular"   # ← Key flag

EXIT_CODE=$?

echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ RETRY SUCCESS"
else
    echo "❌ RETRY FAILED"
fi
