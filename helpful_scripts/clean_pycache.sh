b#!/usr/bin/env bash
set -euo pipefail

echo "Cleaning Python bytecode artifacts..."

count_dirs=$(find . -type d -name "__pycache__" -not -path "./.venv/*" | wc -l | tr -d ' ')
count_pyc=$(find . -type f -name "*.pyc" -not -path "./.venv/*" | wc -l | tr -d ' ')
count_pyo=$(find . -type f -name "*.pyo" -not -path "./.venv/*" | wc -l | tr -d ' ')

find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -not -path "./.venv/*" -delete 2>/dev/null || true

echo "Removed: ${count_dirs} __pycache__ dirs, ${count_pyc} .pyc files, ${count_pyo} .pyo files"
echo "Done."
