#!/usr/bin/env bash
# Set up Python environment for Mamba development
set -e
python -m pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
pip install transformers pytest

