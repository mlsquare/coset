#!/bin/bash

# CoSet Virtual Environment Setup Script
# This script creates a virtual environment and installs all dependencies

echo "🚀 Setting up CoSet Virtual Environment"
echo "======================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📋 Python version: $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv coset_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source coset_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing dependencies..."
pip install numpy scipy pytest pytest-cov black flake8 mypy

# Set environment variable for OpenMP
echo "🔧 Setting up environment variables..."
export KMP_DUPLICATE_LIB_OK=TRUE

# Create activation script
echo "📝 Creating activation script..."
cat > activate_coset.sh << 'EOF'
#!/bin/bash
# CoSet Environment Activation Script

echo "🚀 Activating CoSet Environment"
echo "==============================="

# Activate virtual environment
source coset_env/bin/activate

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "✅ Environment activated!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "🎯 Ready to run CoSet examples!"
echo "   • python quick_test.py"
echo "   • python simple_example.py"
echo "   • python examples/mlp_example.py"
echo ""
EOF

chmod +x activate_coset.sh

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🎯 To activate the environment, run:"
echo "   source activate_coset.sh"
echo ""
echo "📋 Available examples:"
echo "   • python quick_test.py          (Quick test - 30 seconds)"
echo "   • python simple_example.py      (Comprehensive demo - 2-3 minutes)"
echo "   • python examples/mlp_example.py (Full example with training)"
echo ""
echo "🧪 To run tests:"
echo "   python -m pytest tests/ -v"
echo ""
