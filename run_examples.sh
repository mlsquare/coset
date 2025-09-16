#!/bin/bash

# CoSet Examples Runner
# This script provides easy commands to run all CoSet examples

echo "🚀 CoSet Examples Runner"
echo "========================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated."
    echo "   Run: source activate_coset.sh"
    echo "   Then run this script again."
    exit 1
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo "🐍 Python: $(which python)"
echo "📦 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Function to run examples
run_example() {
    local name="$1"
    local command="$2"
    local description="$3"
    
    echo "🎯 $name"
    echo "   $description"
    echo "   Command: $command"
    echo ""
    read -p "   Press Enter to run (or Ctrl+C to skip): "
    echo "   Running..."
    echo ""
    
    eval "$command"
    
    echo ""
    echo "✅ $name completed!"
    echo "=========================================="
    echo ""
}

# Main menu
while true; do
    echo "📋 Available Examples:"
    echo "1. Quick Test (30 seconds)"
    echo "2. Comprehensive Demo (2-3 minutes)"
    echo "3. Full MLP Example (3-5 minutes)"
    echo "4. Run All Tests"
    echo "5. Preview Examples (no computation)"
    echo "6. Exit"
    echo ""
    read -p "Choose an option (1-6): " choice
    
    case $choice in
        1)
            run_example "Quick Test" \
                       "python quick_test.py" \
                       "Basic functionality test - verifies all components work"
            ;;
        2)
            run_example "Comprehensive Demo" \
                       "python simple_example.py" \
                       "4 demos: quantization, MLP training, compression, performance"
            ;;
        3)
            run_example "Full MLP Example" \
                       "python examples/mlp_example.py" \
                       "Complete training workflow with distributed training simulation"
            ;;
        4)
            run_example "Test Suite" \
                       "python -m pytest tests/ -v" \
                       "34 comprehensive test cases covering all functionality"
            ;;
        5)
            run_example "Preview Examples" \
                       "python preview_examples.py" \
                       "Shows what each example will do (no computation)"
            ;;
        6)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid option. Please choose 1-6."
            echo ""
            ;;
    esac
done
