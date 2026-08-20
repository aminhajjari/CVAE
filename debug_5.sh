#!/bin/bash

#SBATCH --account=def-arashmoh
#SBATCH --job-name=AIV2I_DEBUG
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

#SBATCH --output=/home/gkianfar/scratch/Amin/AI/outputs/logs/AIdebug_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/AI/outputs/logs/AIdebug_%A.err


# ============================================================
# Paths
# ============================================================

PROJECT_DIR="/home/gkianfar/scratch/Amin/ICC"

CODE_DIR="/home/gkianfar/scratch/Amin/AI/CVAE"

DEBUG_DATA="/home/gkianfar/scratch/Amin/ICC/debug_data"

VENV_PATH="$PROJECT_DIR/venvMsc"

BATCH_SCRIPT="$CODE_DIR/run_all_datasets.py"

MAIN_SCRIPT="$CODE_DIR/main.py"

OUTPUT="/home/gkianfar/scratch/Amin/AI/outputs"

RESULTS_BASE="$OUTPUT"

JOB_LOGS_DIR="$OUTPUT/logs"

TIMEOUT=14400


# ============================================================
# Setup
# ============================================================

mkdir -p "$OUTPUT"
mkdir -p "$JOB_LOGS_DIR"

echo "=========================================="
echo "V2I DEBUG RUN - 5 DATASETS"
echo "=========================================="

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

echo "Project:"
echo "$PROJECT_DIR"

echo "Code:"
echo "$CODE_DIR"

echo "Debug data:"
echo "$DEBUG_DATA"

echo "Virtual environment:"
echo "$VENV_PATH"

echo "Batch script:"
echo "$BATCH_SCRIPT"

echo "Main script:"
echo "$MAIN_SCRIPT"

echo ""


# ============================================================
# Check debug datasets
# ============================================================

echo "=========================================="
echo "DEBUG DATASETS"
echo "=========================================="

if [ ! -d "$DEBUG_DATA" ]; then
    echo "ERROR: Debug data directory does not exist:"
    echo "$DEBUG_DATA"
    exit 1
fi

ls -lah "$DEBUG_DATA"

echo ""


# ============================================================
# Load environment
# ============================================================

module purge

module load StdEnv/2023
module load python/3.11
module load cuda/12.2


# ============================================================
# Activate virtual environment
# ============================================================

source "$VENV_PATH/bin/activate"

echo "=========================================="
echo "PYTHON ENVIRONMENT"
echo "=========================================="

which python
python --version

echo ""
echo "Virtual environment:"
echo "$VIRTUAL_ENV"

echo ""


# ============================================================
# Move to code directory
# ============================================================

cd "$CODE_DIR" || {
    echo "ERROR: Cannot cd to $CODE_DIR"
    exit 1
}


# ============================================================
# Check GPU
# ============================================================

echo "=========================================="
echo "GPU"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""


# ============================================================
# Check PyTorch
# ============================================================

echo "=========================================="
echo "PYTORCH"
echo "=========================================="

python -c "
import torch

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo ""


# ============================================================
# Check required scripts
# ============================================================

echo "=========================================="
echo "SCRIPT CHECK"
echo "=========================================="

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "ERROR: Batch script not found:"
    echo "$BATCH_SCRIPT"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "ERROR: Main script not found:"
    echo "$MAIN_SCRIPT"
    exit 1
fi

echo "Batch script:"
echo "$BATCH_SCRIPT"

echo "Main script:"
echo "$MAIN_SCRIPT"

echo ""


# ============================================================
# Run 5 datasets
# ============================================================

echo "=========================================="
echo "STARTING DEBUG RUN"
echo "=========================================="

echo "Start time: $(date)"
echo ""

python "$BATCH_SCRIPT" \
    --datasets_dir "$DEBUG_DATA" \
    --output_base "$OUTPUT" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
    --timeout "$TIMEOUT"

EXIT_CODE=$?


# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "DEBUG RUN COMPLETE"
echo "=========================================="

echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results: $OUTPUT"

echo "=========================================="

exit $EXIT_CODE
