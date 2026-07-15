#!/usr/bin/env bash
# X-AnyLabeling / TrainLens Development Launcher for Linux/macOS
# Automatically installs uv, sets up Python environment, and launches the application

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "X-AnyLabeling Development Launcher"
echo "========================================"
echo ""

# ============================================
# Step 1: Check for uv installation
# ============================================
echo "[1/6] Checking for uv package manager..."
if command -v uv &> /dev/null; then
    echo "[OK] uv is already installed"
else
    echo "[INFO] uv not found. Installing uv..."
    echo ""

    # Try to install uv using pip
    if python3 -m pip install --upgrade uv &> /dev/null; then
        echo "[OK] uv installed successfully via pip"
    else
        # If pip install failed, try using curl installer
        echo "[INFO] Attempting to install uv via curl installer..."
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            echo "[OK] uv installed successfully"
            # Source the environment to get uv in PATH
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            echo "[ERROR] Failed to install uv"
            echo "Please install uv manually: https://github.com/astral-sh/uv"
            exit 1
        fi
    fi
    echo ""
fi

# ============================================
# Step 2: Check for .venv directory
# ============================================
echo "[2/6] Checking for virtual environment..."
if [ -d ".venv" ]; then
    echo "[OK] Virtual environment already exists"
else
    echo "[INFO] Creating virtual environment..."
    uv venv .venv
    echo "[OK] Virtual environment created"
    echo ""
fi

# ============================================
# Step 3: Detect GPU and determine extra
# ============================================
echo "[3/6] Detecting GPU..."

EXTRA="cpu"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "[OK] NVIDIA GPU detected"
        EXTRA="gpu"
    else
        echo "[INFO] nvidia-smi found but no GPU detected, using CPU mode"
    fi
else
    echo "[INFO] No NVIDIA GPU detected, using CPU mode"
fi

# ============================================
# Step 4: Install dependencies
# ============================================
echo "[4/6] Installing dependencies with extra: $EXTRA..."
echo ""

# Verify extra exists in pyproject.toml
if ! grep -q "\\[$EXTRA\\]" pyproject.toml; then
    echo "[WARNING] Extra '$EXTRA' not found in pyproject.toml, falling back to 'cpu'"
    EXTRA="cpu"
fi

# Install with uv
if uv pip install -e ".[$EXTRA,dev]"; then
    echo ""
    echo "[OK] Dependencies installed successfully"
    echo ""
else
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

# ============================================
# Step 5: Verify installation
# ============================================
echo "[5/6] Verifying installation..."

# Activate virtual environment
source .venv/bin/activate

# Check if anylabeling module is importable
if python -c "import anylabeling; print('Version:', anylabeling.app_info.__version__)" &> /dev/null; then
    echo "[OK] Installation verified"
    echo ""
else
    echo "[ERROR] Failed to import anylabeling module"
    echo "Please check the installation"
    exit 1
fi

# ============================================
# Step 6: Launch application
# ============================================
echo "[6/6] Launching X-AnyLabeling..."
echo ""
echo "========================================"
echo ""

# Launch the application with any passed arguments
python -m anylabeling.app "$@"

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "[ERROR] Application exited with error code: $EXIT_CODE"
    echo "========================================"
    exit $EXIT_CODE
fi

echo ""
echo "========================================"
echo "Application closed successfully"
echo "========================================"
exit 0
