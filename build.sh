#!/bin/bash
# ============================================================================
# Unfold Portable - Linux/macOS Build Script
# ============================================================================
# This script builds Unfold as a standalone executable.
#
# Prerequisites:
#   - Python 3.11+ installed
#   - pip install pyinstaller
#
# Usage:
#   ./build.sh           - Build the executable
#   ./build.sh clean     - Clean build artifacts
#   ./build.sh full      - Clean and rebuild everything
# ============================================================================

set -e

echo ""
echo "============================================================"
echo "  Unfold Portable - Build Script"
echo "============================================================"
echo ""

# Function to clean build artifacts
clean() {
    echo "Cleaning build artifacts..."
    rm -rf build dist __pycache__ version_info.txt
    echo "Done."
}

# Handle command line arguments
if [ "$1" = "clean" ]; then
    clean
    exit 0
fi

if [ "$1" = "full" ]; then
    clean
fi

# Check Python installation
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi
python3 --version

# Check/Install PyInstaller
echo "[2/5] Checking PyInstaller..."
if ! pip3 show pyinstaller &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    pip3 install pyinstaller
fi

# Install dependencies
echo "[3/5] Installing dependencies..."
pip3 install -r requirements_portable.txt --quiet 2>/dev/null || true

# Build executable
echo "[4/5] Building executable with PyInstaller..."
pyinstaller unfold.spec --noconfirm

# Create package structure
echo "[5/5] Creating portable package structure..."
mkdir -p dist/Unfold_Portable/unfold_data/{graphs,cache}

if [ -f "dist/Unfold" ]; then
    cp dist/Unfold dist/Unfold_Portable/
elif [ -d "dist/Unfold" ]; then
    cp -r dist/Unfold/* dist/Unfold_Portable/
fi

# Create launcher script for Unix
cat > dist/Unfold_Portable/start_unfold.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
./Unfold
EOF
chmod +x dist/Unfold_Portable/start_unfold.sh

# Create default config
cat > dist/Unfold_Portable/unfold_data/config.json << 'EOF'
{
  "host": "127.0.0.1",
  "port": 8080,
  "open_browser": true,
  "debug": false,
  "ollama_host": "http://localhost:11434",
  "llm_provider": "ollama",
  "llm_model": "llama3.2"
}
EOF

# Create portable marker
echo "Unfold Portable Edition" > dist/Unfold_Portable/portable.marker

echo ""
echo "============================================================"
echo "  BUILD COMPLETE!"
echo "============================================================"
echo ""
echo "Output location: dist/Unfold_Portable/"
echo ""
echo "To run:"
echo "  cd dist/Unfold_Portable && ./start_unfold.sh"
echo ""
echo "For LLM-powered extraction, install Ollama:"
echo "  curl -fsSL https://ollama.ai/install.sh | sh"
echo "  ollama pull llama3.2"
echo ""
