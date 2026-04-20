#!/bin/bash
echo "=== Starting D7047E environment ==="

# Install git (needed for version control)
apt-get update -qq && apt-get install -y git

# Install Python packages from requirements.txt
echo "Installing Python packages..."
pip install -r requirements.txt

echo "✅ Git and packages installed successfully!"
echo ""
echo "Next step: Run 'wandb-login' and paste your own API key"
