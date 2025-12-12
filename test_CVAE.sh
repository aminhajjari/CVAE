#!/bin/bash

#=======================================================================
# TEST SLURM SCRIPT - Real CVAE + ADOPT + Dual SHAP
#=======================================================================
# Quick debug script for:
#   - Real CVAE (image+tab) with KL
#   - ADOPT optimizer (decoupled weight decay)
#   - Dual SHAP interpretability outputs
#   - Robustness on small tabular datasets
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=CVAE_ADOPT_DEBUG
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/CVAE/job_logs/cvae_debug_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/CVAE/job_logs/cvae_debug_%j.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=END,FAIL

#=======================================================================
# Configuration
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
TAB2IMG_DIR="$PROJECT_DIR/Tab2img"
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"
MAIN_SCRIPT="$TAB2IMG_DIR/run_vif.py"

# Small datasets for fast CVAE+SHAP debugging
TEST_DATASETS=(
    "balance-scale"
    "tic-tac-toe"
    "blood-transfusion-service-center"
)

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "REAL CVAE + ADOPT + DUAL SHAP DEBUG"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Testing ${#TEST_DATASETS[@]} datasets"
echo "Optimizer: ADOPT (decoupled), WD=1e-4"
echo "=========================================="
echo ""

#=======================================================================
# GPU Information
#=======================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

#=======================================================================
# Load Environment
#=======================================================================
echo "Loading environment..."
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

echo "Activating virtual environment..."
source "$VENV_PATH"

echo ""
echo "Python environment:"
python --version
python - << 'EOF'
import torch, numpy, pandas
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF

echo ""
echo "Checking dependencies (SHAP, ADOPT)..."
python << 'PYCHECK'
try:
    import shap
    print(f"✓ SHAP installed: version {shap.__version__}")
except ImportError:
    print("✗ SHAP not installed - script will fail!")
    exit(1)

try:
    from adopt import ADOPT
    print("✓ ADOPT optimizer available")
except Exception as e:
    print(f"✗ ADOPT import failed: {e}")
    exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Required Python deps missing (SHAP and/or ADOPT)."
    exit 1
fi

echo ""
echo "Environment ready"
echo ""

#=======================================================================
# Test Each Dataset
#=======================================================================
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_TIME=0

declare -a DATASET_NAMES
declare -a TAB_ACCURACIES
declare -a IMG_ACCURACIES
declare -a AUC_VALUES

echo "=========================================="
echo "STARTING TEST RUNS (CVAE + ADOPT)"
echo "=========================================="

for dataset in "${TEST_DATASETS[@]}"; do
    echo ""
    echo "======================================"
    echo "Testing dataset: $dataset"
    echo "======================================"

    DATASET_PATH=$(find "$DATASETS_DIR" -type d -name "$dataset" | head -1)

    if [ -z "$DATASET_PATH" ]; then
        echo "✗ Dataset not found: $dataset"
        FAIL_COUNT=$((FAIL_COUNT+1))
        continue
    fi

    DATA_FILE=$(find "$DATASET_PATH" -type f \( -name "*.arff" -o -name "*.csv" -o -name "*.data" \) | head -1)

    if [ -z "$DATA_FILE" ]; then
        echo "✗ No usable data file found for: $dataset"
        FAIL_COUNT=$((FAIL_COUNT+1))
        continue
    fi

    echo "Dataset path: $DATASET_PATH"
    echo "Data file:   $DATA_FILE"
    echo ""

    START_TIME=$(date +%s)

    LOG_FILE="/tmp/test_${dataset}_${SLURM_JOB_ID}.log"

    python "$MAIN_SCRIPT" \
        --data "$DATA_FILE" \
        --num_images 5 \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    TOTAL_TIME=$((TOTAL_TIME + ELAPSED))

    echo ""
    echo "--------------------------------------"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ SUCCESS in ${ELAPSED}s"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Extract JSON summary that run_vif.py prints (RESULTS_JSON_START/END)
        SUMMARY_JSON=$(awk '/RESULTS_JSON_START/{flag=1;next}/RESULTS_JSON_END/{flag=0}flag' "$LOG_FILE")

        if [ -n "$SUMMARY_JSON" ]; then
            TAB_ACC=$(python - << EOF
import json,sys
data=json.loads('''$SUMMARY_JSON''')
print(data.get("best_accuracy", "NA"))
EOF
)
            AUC_VAL=$(python - << EOF
import json,sys
data=json.loads('''$SUMMARY_JSON''')
print(data.get("best_auc", "NA"))
EOF
)
            # For image accuracy you currently log "Img Acc" in epoch prints;
            # parse the last occurrence as a proxy.
            IMG_ACC=$(grep "Img Acc" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+' | tail -1)

            DATASET_NAMES+=("$dataset")
            TAB_ACCURACIES+=("$TAB_ACC")
            IMG_ACCURACIES+=("${IMG_ACC:-NA}")
            AUC_VALUES+=("$AUC_VAL")

            echo "   📊 Summary: Tab Acc=${TAB_ACC}% | Img Acc=${IMG_ACC:-NA}% | AUC=${AUC_VAL}"
        else
            echo "   ⚠ No RESULTS_JSON block found in log (cannot parse metrics)."
        fi

        # Check for Dual SHAP outputs
        INTERP_DIR="$dataset/dual_shap_interpretability"

        if [ -d "$INTERP_DIR" ]; then
            echo ""
            echo "Dual SHAP files generated:"
            CSV_COUNT=$(find "$INTERP_DIR" -name "*.csv" | wc -l)
            PNG_COUNT=$(find "$INTERP_DIR" -name "*.png" | wc -l)
            NPY_COUNT=$(find "$INTERP_DIR" -name "*.npy" | wc -l)
            TXT_COUNT=$(find "$INTERP_DIR" -name "*.txt" | wc -l)

            echo "  CSV files: $CSV_COUNT"
            echo "  PNG plots: $PNG_COUNT"
            echo "  NPY arrays: $NPY_COUNT"
            echo "  Text reports: $TXT_COUNT"
        else
            echo "⚠ Warning: Interpretability directory not found: $INTERP_DIR"
        fi

    else
        echo "✗ FAILED (exit code: $EXIT_CODE) after ${ELAPSED}s"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        echo "Error preview (last 30 lines):"
        tail -30 "$LOG_FILE"
    fi

    echo "======================================"
done

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "CVAE + ADOPT DEBUG SUMMARY"
echo "=========================================="
echo "Datasets tested: ${#TEST_DATASETS[@]}"
echo "✓ Success: $SUCCESS_COUNT"
echo "✗ Failed:  $FAIL_COUNT"
echo "Total time: ${TOTAL_TIME}s ($(($TOTAL_TIME/60))m)"
echo ""

if [ ${#DATASET_NAMES[@]} -gt 0 ]; then
    echo "=========================================="
    echo "ACCURACY / AUC RESULTS"
    echo "=========================================="
    printf "%-40s %10s %10s %10s\n" "Dataset" "Tabular" "Image" "AUC"
    echo "----------------------------------------"
    for i in "${!DATASET_NAMES[@]}"; do
        printf "%-40s %9s%% %9s%% %9s\n" \
            "${DATASET_NAMES[$i]}" \
            "${TAB_ACCURACIES[$i]}" \
            "${IMG_ACCURACIES[$i]}" \
            "${AUC_VALUES[$i]}"
    done
    echo "=========================================="
    echo ""
fi

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Interpretability outputs are in:"
    for dataset in "${TEST_DATASETS[@]}"; do
        if [ -d "$dataset/dual_shap_interpretability" ]; then
            echo "  • $dataset/dual_shap_interpretability/"
        fi
    done
fi

echo ""
echo "Finished: $(date)"
echo "=========================================="

# Save results for later comparison
if [ ${#DATASET_NAMES[@]} -gt 0 ]; then
    RESULTS_FILE="test_results_cvae_adopt_${SLURM_JOB_ID}.txt"
    {
        echo "TEST RESULTS - Real CVAE + ADOPT + Dual SHAP"
        echo "Date: $(date)"
        echo ""
        printf "%-40s %10s %10s %10s\n" "Dataset" "Tabular" "Image" "AUC"
        echo
