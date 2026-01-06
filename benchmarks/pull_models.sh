#!/bin/bash
# =============================================================================
# Echo Brain Benchmark - Model Puller
# =============================================================================
# Downloads all candidate models for benchmarking.
# Run this before benchmark.py to ensure all models are available.
#
# Usage:
#   ./pull_models.sh           # Pull all models
#   ./pull_models.sh --minimal # Pull only recommended subset
# =============================================================================

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧠 Echo Brain Model Puller"
echo "=========================="
echo "Ollama host: $OLLAMA_HOST"
echo ""

# Check Ollama is running
if ! curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo -e "${RED}❌ Ollama not responding at $OLLAMA_HOST${NC}"
    echo "Start Ollama first: ollama serve"
    exit 1
fi

# =============================================================================
# MODEL LISTS
# =============================================================================

# Classification models (fast, small)
CLASSIFICATION_MODELS=(
    "qwen2.5:1.5b"
    "qwen2.5:3b"
    "llama3.2:3b"
    "phi3:mini"
)

# Coding models (your primary use case)
CODING_MODELS=(
    "qwen2.5-coder:7b"
    "qwen2.5-coder:14b"
    "deepseek-coder-v2:16b"
    "codellama:13b"
)

# Reasoning models
REASONING_MODELS=(
    "deepseek-r1:8b"
    "qwen2.5:14b"
    "llama3.1:8b"
    "phi3:14b"
)

# General models
GENERAL_MODELS=(
    "llama3.1:8b"
    "qwen2.5:7b"
    "mistral:7b"
)

# Minimal set (recommended subset for quick testing)
MINIMAL_MODELS=(
    "qwen2.5:3b"           # Classification
    "qwen2.5-coder:7b"     # Coding
    "deepseek-r1:8b"       # Reasoning
    "llama3.1:8b"          # General
)

# =============================================================================
# FUNCTIONS
# =============================================================================

pull_model() {
    local model=$1
    echo -n "  Pulling $model... "

    # Check if already exists
    if ollama list 2>/dev/null | grep -q "^$model"; then
        echo -e "${GREEN}already exists${NC}"
        return 0
    fi

    # Pull the model
    if ollama pull "$model" > /dev/null 2>&1; then
        echo -e "${GREEN}done${NC}"
        return 0
    else
        echo -e "${RED}failed${NC}"
        return 1
    fi
}

pull_category() {
    local category=$1
    shift
    local models=("$@")

    echo ""
    echo -e "${YELLOW}$category Models:${NC}"

    local success=0
    local failed=0

    for model in "${models[@]}"; do
        if pull_model "$model"; then
            ((success++))
        else
            ((failed++))
        fi
    done

    echo "  ✓ $success pulled, ✗ $failed failed"
}

show_vram_estimate() {
    echo ""
    echo "VRAM Estimates (approximate):"
    echo "─────────────────────────────────────────"
    echo "  1.5B models:  ~1.5 GB"
    echo "  3B models:    ~2.5 GB"
    echo "  7B models:    ~5 GB"
    echo "  8B models:    ~6 GB"
    echo "  13B models:   ~9 GB"
    echo "  14B models:   ~10 GB"
    echo "  16B models:   ~11 GB (tight on 12GB card)"
    echo ""
    echo "Your 3060 (12GB): Can run up to 14B comfortably"
    echo "                  16B models may need quantization"
}

# =============================================================================
# MAIN
# =============================================================================

if [[ "$1" == "--minimal" ]]; then
    echo "Pulling MINIMAL model set (recommended for quick testing)"
    echo ""

    for model in "${MINIMAL_MODELS[@]}"; do
        pull_model "$model"
    done

    show_vram_estimate

    echo ""
    echo -e "${GREEN}✅ Minimal set ready!${NC}"
    echo "Run: python benchmark.py --models ${MINIMAL_MODELS[*]}"

elif [[ "$1" == "--classification" ]]; then
    pull_category "Classification" "${CLASSIFICATION_MODELS[@]}"

elif [[ "$1" == "--coding" ]]; then
    pull_category "Coding" "${CODING_MODELS[@]}"

elif [[ "$1" == "--reasoning" ]]; then
    pull_category "Reasoning" "${REASONING_MODELS[@]}"

elif [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  (none)           Pull all benchmark models"
    echo "  --minimal        Pull recommended subset only"
    echo "  --classification Pull classification models only"
    echo "  --coding         Pull coding models only"
    echo "  --reasoning      Pull reasoning models only"
    echo "  --help           Show this help"
    echo ""
    echo "Models by category:"
    echo ""
    echo "Classification: ${CLASSIFICATION_MODELS[*]}"
    echo "Coding:         ${CODING_MODELS[*]}"
    echo "Reasoning:      ${REASONING_MODELS[*]}"
    echo "General:        ${GENERAL_MODELS[*]}"
    echo ""
    echo "Minimal (quick): ${MINIMAL_MODELS[*]}"

else
    echo "Pulling ALL benchmark models"
    echo "This may take a while and use significant disk space (~50GB+)"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    pull_category "Classification" "${CLASSIFICATION_MODELS[@]}"
    pull_category "Coding" "${CODING_MODELS[@]}"
    pull_category "Reasoning" "${REASONING_MODELS[@]}"
    pull_category "General" "${GENERAL_MODELS[@]}"

    show_vram_estimate

    echo ""
    echo -e "${GREEN}✅ All models ready!${NC}"
    echo "Run: python benchmark.py"
fi

echo ""
echo "Current models:"
ollama list