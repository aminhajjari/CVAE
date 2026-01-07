#!/bin/bash

#=======================================================================
# PRODUCTION SLURM - CVAE on All Tabular Datasets
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=CVAE_Production
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/CVAE/job_logs/production_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/CVAE/job_logs/production_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration - EXACT paths from your system
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
CVAE_DIR="$PROJECT_DIR/CVAE"
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"
BATCH_SCRIPT="$CVAE_DIR/run_all_datasets.py"
MAIN_SCRIPT="$CVAE_DIR/run_vif.py"
RESULTS_BASE="$CVAE_DIR/results"
JOB_LOGS_DIR="$CVAE_DIR/job_logs"

TIMEOUT_DEFAULT=7200  # 2 hours per dataset

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "🧬 CVAE PRODUCTION RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo ""
echo "Configuration:"
echo "  Model: CVAE (Conditional Variational Autoencoder)"
echo "  Optimizer: ADOPT (decoupled weight decay)"
echo "  Loss: Reconstruction + 2×Classification + KL"
echo "  Dual SHAP: Enabled (9 files/dataset)"
echo "  Timeout: 2 hours per dataset"
echo "=========================================="
echo ""

#=======================================================================
# GPU Information
#=======================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

#=======================================================================
# Setup
#=======================================================================
echo "Creating directories..."
mkdir -p "$JOB_LOGS_DIR"
mkdir -p "$RESULTS_BASE"
echo "✅ Directories ready"
echo ""

#=======================================================================
# Verify Files
#=======================================================================
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ ERROR: Batch script not found: $BATCH_SCRIPT"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main script not found: $MAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual env not found: $VENV_PATH"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"
echo ""

#=======================================================================
# Load Environment
#=======================================================================
echo "Loading modules..."
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
echo "✅ Modules loaded"
echo ""

echo "Activating virtual environment..."
source "$VENV_PATH"
echo "✅ Virtual environment active"
echo ""

echo "Python environment:"
python --version
python -c "
import torch, shap
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'SHAP: {shap.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Check ADOPT
try:
    from adopt import ADOPT
    print('ADOPT: ✓ Available')
except Exception as e:
    print(f'ADOPT: ✗ Error - {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Environment check failed!"
    exit 1
fi

echo "✅ Environment ready"
echo ""

#=======================================================================
# Verify CVAE Model
#=======================================================================
echo "Verifying CVAE configuration..."
if grep -q "class CVAEWithTabEmbedding" "$MAIN_SCRIPT"; then
    echo "✅ CVAE model found in run_vif.py"
else
    echo "⚠️  WARNING: CVAEWithTabEmbedding not found"
    echo "   Make sure your model class name matches!"
fi

if grep -q "from adopt import ADOPT" "$MAIN_SCRIPT"; then
    echo "✅ ADOPT optimizer configured"
else
    echo "⚠️  WARNING: ADOPT import not found"
fi
echo ""

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo "=========================================="
echo "🚀 STARTING BATCH PROCESSING"
echo "=========================================="
echo "Using run_all_datasets.py (CVAE version)"
echo ""
echo "Running command:"
echo "python $BATCH_SCRIPT \\"
echo "  --datasets_dir $DATASETS_DIR \\"
echo "  --output_base $RESULTS_BASE \\"
echo "  --job_id $SLURM_JOB_ID \\"
echo "  --script_path $MAIN_SCRIPT \\"
echo "  --timeout $TIMEOUT_DEFAULT"
echo ""
echo "=========================================="
echo ""

# Run the batch processor
python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT"

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "CVAE PRODUCTION RUN COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    RESULT_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "*_JOB${SLURM_JOB_ID}" | head -1)
    
    echo "✅ SUCCESS!"
    echo ""
    echo "📂 Results location:"
    echo "    $RESULT_DIR/"
    echo ""
    echo "📊 Files generated:"
    echo "    ├── csv/"
    echo "    │   ├── results_summary.csv"
    echo "    │   ├── statistics.csv"
    echo "    │   └── interpretability_summary.csv"
    echo "    ├── latex/"
    echo "    │   └── results_latex.txt"
    echo "    ├── logs/"
    echo "    │   └── results.jsonl"
    echo "    └── interpretability/"
    echo "        └── [dataset]/dual_shap_interpretability/"
    echo ""
    
    if [ -d "$RESULT_DIR/interpretability" ]; then
        INTERP_COUNT=$(find "$RESULT_DIR/interpretability" -type d -name "dual_shap_interpretability" | wc -l)
        echo "🔍 Interpretability: $INTERP_COUNT/$DATASET_COUNT datasets"
    fi
    
    if [ -f "$RESULT_DIR/csv/statistics.csv" ]; then
        echo ""
        echo "📊 Quick Statistics:"
        head -5 "$RESULT_DIR/csv/statistics.csv" | column -t -s','
    fi
    
    echo ""
    echo "📧 Completion email sent to: aminhajjr@gmail.com"
    echo "🎉 All $DATASET_COUNT datasets processed with CVAE!"
    
else
    echo "⚠️  Some datasets may have failed"
    echo ""
    echo "Check logs:"
    echo "    Output: $JOB_LOGS_DIR/production_${SLURM_JOB_ID}.out"
    echo "    Error:  $JOB_LOGS_DIR/production_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit $EXIT_CODE
