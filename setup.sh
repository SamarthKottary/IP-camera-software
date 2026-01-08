#!/bin/bash

echo "Installing System Dependencies..."

# 1. Update Apt
sudo apt-get update

# 2. Install OpenCV and Clang (Required for the 'opencv' crate)
sudo apt-get install -y libopencv-dev clang libclang-dev pkg-config build-essential git

# 3. Print success
echo "------------------------------------------------"
echo "System requirements installed."
echo "You can now run: cargo build --release"
echo "------------------------------------------------"
