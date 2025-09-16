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
