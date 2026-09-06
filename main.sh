#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - CVAE / V2I - All Datasets
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=V2I
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

# SLURM logs
#SBATCH --output=/home/gkianfar/scratch/Amin/AI/outputs/logs/ALLCVAE_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/AI/outputs/logs/ALLCVAE_%A.err


#=======================================================================
# Configuration
#=======================================================================

# Project / code
PROJECT_DIR="/home/gkianfar/scratch/Amin/AI"
TAB2IMG_DIR="$PROJECT_DIR/CVAE"

# Datasets
DATASETS_DIR="/home/gkianfar/scratch/Amin/ICC/Unzippeddata/CSV"

# Python virtual environment
VENV_PATH="/home/gkianfar/scratch/Amin/ICC/venvMsc/bin/activate"

# Python scripts
BATCH_SCRIPT="$TAB2IMG_DIR/run_all_datasets.py"
MAIN_SCRIPT="$TAB2IMG_DIR/main.py"
# 28800 seconds = 8 hours
TIMEOUT_DEFAULT=28800


#=======================================================================
# Job Information
#=======================================================================

echo "=========================================="
echo "TABLE2IMAGE-CVAE PRODUCTION RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo ""
echo "Project dir:  $PROJECT_DIR"
echo "Code dir:     $TAB2IMG_DIR"
echo "Datasets dir: $DATASETS_DIR"
echo "Output dir:   $RESULTS_BASE"
echo "Image dir:    $IMAGE_OUTPUT_DIR"
echo "Logs dir:     $JOB_LOGS_DIR"
echo ""
echo "Configuration:"
echo "  - GPU: H100"
echo "  - CPUs: 8 cores"
echo "  - Memory: 64GB"
echo "  - SLURM walltime: 96 hours"
echo "  - Dataset timeout: 8 hours"
echo "  - Weight Decay: 1e-4 (AdamW)"
echo "=========================================="
echo ""


#=======================================================================
# Create Required Directories
#=======================================================================

echo "Creating directories..."

mkdir -p "$RESULTS_BASE"
mkdir -p "$IMAGE_OUTPUT_DIR"
mkdir -p "$JOB_LOGS_DIR"

echo "Directories:"
echo "  Results: $RESULTS_BASE"
echo "  Images:  $IMAGE_OUTPUT_DIR"
echo "  Logs:    $JOB_LOGS_DIR"
echo ""

echo "Directory creation complete."
echo ""


#=======================================================================
# GPU Information
#=======================================================================

echo "=========================================="
echo "GPU INFORMATION"
echo "=========================================="

if command -v nvidia-smi >/dev/null 2>&1; then

    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader

else

    echo "WARNING: nvidia-smi not found."

fi

echo ""


#=======================================================================
# Verify Files
#=======================================================================

echo "=========================================="
echo "VERIFYING FILES"
echo "=========================================="

#-----------------------------------------------------------------------
# Dataset directory
#-----------------------------------------------------------------------

if [ ! -d "$DATASETS_DIR" ]; then

    echo "ERROR: Dataset directory not found:"
    echo "  $DATASETS_DIR"
    exit 1

fi

echo "OK: Dataset directory found."


#-----------------------------------------------------------------------
# Batch script
#-----------------------------------------------------------------------

if [ ! -f "$BATCH_SCRIPT" ]; then

    echo "ERROR: Batch script not found:"
    echo "  $BATCH_SCRIPT"
    exit 1

fi

echo "OK: Batch script found:"
echo "  $BATCH_SCRIPT"


#-----------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------

if [ ! -f "$MAIN_SCRIPT" ]; then

    echo "ERROR: Main script not found:"
    echo "  $MAIN_SCRIPT"
    exit 1

fi

echo "OK: Main script found:"
echo "  $MAIN_SCRIPT"


#-----------------------------------------------------------------------
# Virtual environment
#-----------------------------------------------------------------------

if [ ! -f "$VENV_PATH" ]; then

    echo "ERROR: Virtual environment not found:"
    echo "  $VENV_PATH"
    exit 1

fi

echo "OK: Virtual environment found:"
echo "  $VENV_PATH"

echo ""


#=======================================================================
# Count Datasets
#=======================================================================

DATASET_COUNT=$(find "$DATASETS_DIR" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d | wc -l)

echo "Found $DATASET_COUNT dataset folders."

if [ "$DATASET_COUNT" -eq 0 ]; then

    echo "ERROR: No dataset folders found."
    exit 1

fi

echo ""


#=======================================================================
# Load Environment
#=======================================================================

echo "=========================================="
echo "LOADING ENVIRONMENT"
echo "=========================================="

module purge

module load StdEnv/2023
module load python/3.11
module load cuda/12.2

echo "Modules loaded:"
module list 2>&1

echo ""


#=======================================================================
# Activate Virtual Environment
#=======================================================================

echo "Activating virtual environment..."

source "$VENV_PATH"

if [ $? -ne 0 ]; then

    echo "ERROR: Failed to activate virtual environment."
    exit 1

fi

echo "Virtual environment activated."
echo ""


#=======================================================================
# Python Environment Check
#=======================================================================

echo "=========================================="
echo "PYTHON ENVIRONMENT"
echo "=========================================="

echo "Python:"
which python

echo ""

echo "Python version:"
python --version

echo ""

echo "Python executable:"
python -c "import sys; print(sys.executable)"

echo ""

echo "PyTorch / CUDA check:"

python -c "
import torch

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('PyTorch CUDA version:', torch.version.cuda)

if torch.cuda.is_available():

    print('GPU count:', torch.cuda.device_count())
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU capability:', torch.cuda.get_device_capability(0))

"

if [ $? -ne 0 ]; then

    echo ""
    echo "ERROR: Python/PyTorch environment check failed."
    exit 1

fi

echo ""
echo "Python environment ready."
echo ""


#=======================================================================
# Verify Weight Decay
#=======================================================================

echo "=========================================="
echo "VERIFYING WEIGHT DECAY"
echo "=========================================="

if grep -q "weight_decay=1e-4" "$MAIN_SCRIPT"; then

    echo "OK: weight_decay=1e-4 found in:"
    echo "  $MAIN_SCRIPT"

else

    echo "WARNING: weight_decay=1e-4 was NOT found in:"
    echo "  $MAIN_SCRIPT"

    echo ""
    echo "Please verify the weight decay configuration manually."

fi

echo ""


#=======================================================================
# Print Final Configuration
#=======================================================================

echo "=========================================="
echo "FINAL CONFIGURATION"
echo "=========================================="

echo "Project:"
echo "  $PROJECT_DIR"

echo ""
echo "Code:"
echo "  $TAB2IMG_DIR"

echo ""
echo "Batch script:"
echo "  $BATCH_SCRIPT"

echo ""
echo "Main script:"
echo "  $MAIN_SCRIPT"

echo ""
echo "Datasets:"
echo "  $DATASETS_DIR"

echo ""
echo "Virtual environment:"
echo "  $VENV_PATH"

echo ""
echo "Results:"
echo "  $RESULTS_BASE"

echo ""
echo "Images:"
echo "  $IMAGE_OUTPUT_DIR"

echo ""
echo "Logs:"
echo "  $JOB_LOGS_DIR"

echo ""
echo "Dataset count:"
echo "  $DATASET_COUNT"

echo ""
echo "Per-dataset timeout:"
echo "  $TIMEOUT_DEFAULT seconds (8 hours)"

echo ""
echo "=========================================="
echo ""


#=======================================================================
# Execute Batch Processing
#=======================================================================

echo "=========================================="
echo "STARTING BATCH PROCESSING"
echo "=========================================="

echo "Command:"
echo ""
echo "python $BATCH_SCRIPT \\"
echo "    --datasets_dir $DATASETS_DIR \\"
echo "    --output_base $RESULTS_BASE \\"
echo "    --job_id $SLURM_JOB_ID \\"
echo "    --script_path $MAIN_SCRIPT \\"
echo "    --timeout $TIMEOUT_DEFAULT \\"
echo "    --skip_existing"

echo ""
echo "=========================================="
echo ""


#-----------------------------------------------------------------------
# Move to code directory
#-----------------------------------------------------------------------

cd "$TAB2IMG_DIR"

if [ $? -ne 0 ]; then

    echo "ERROR: Could not change to code directory:"
    echo "  $TAB2IMG_DIR"
    exit 1

fi


#-----------------------------------------------------------------------
# Run batch processor
#-----------------------------------------------------------------------

python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT" \
    --skip_existing

EXIT_CODE=$?


#=======================================================================
# Final Summary
#=======================================================================

echo ""
echo "=========================================="
echo "PRODUCTION RUN COMPLETE"
echo "=========================================="

echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""


#=======================================================================
# Success
#=======================================================================

if [ "$EXIT_CODE" -eq 0 ]; then

    echo "SUCCESS!"
    echo ""

    echo "Results base directory:"
    echo "  $RESULTS_BASE"
    echo ""

    echo "Image output directory:"
    echo "  $IMAGE_OUTPUT_DIR"
    echo ""

    echo "SLURM logs:"
    echo "  $JOB_LOGS_DIR/ALLCVAE_${SLURM_JOB_ID}.out"
    echo "  $JOB_LOGS_DIR/ALLCVAE_${SLURM_JOB_ID}.err"
    echo ""

    #-------------------------------------------------------------------
    # Find job-specific result directory
    #-------------------------------------------------------------------

    RESULT_DIR=$(find "$RESULTS_BASE" \
        -maxdepth 1 \
        -type d \
        -name "*_JOB${SLURM_JOB_ID}" \
        | head -1)

    if [ -n "$RESULT_DIR" ]; then

        echo "Job result directory:"
        echo "  $RESULT_DIR"
        echo ""

        echo "Generated files/directories:"
        find "$RESULT_DIR" -maxdepth 2 -type f | head -50

    else

        echo "No *_JOB${SLURM_JOB_ID} result directory detected."
        echo "Please inspect:"
        echo "  $RESULTS_BASE"

    fi

    echo ""
    echo "All $DATASET_COUNT datasets processed."

else

    echo "WARNING: Batch processor returned exit code $EXIT_CODE."
    echo ""
    echo "Some datasets may have failed."
    echo ""
    echo "Check SLURM output log:"
    echo "  $JOB_LOGS_DIR/ALLCVAE_${SLURM_JOB_ID}.out"
    echo ""
    echo "Check SLURM error log:"
    echo "  $JOB_LOGS_DIR/ALLCVAE_${SLURM_JOB_ID}.err"
    echo ""
    echo "Check generated results:"
    echo "  $RESULTS_BASE"
    echo ""

fi


#=======================================================================
# End
#=======================================================================

echo "=========================================="
echo "JOB FINISHED"
echo "=========================================="

exit "$EXIT_CODE"
