#!/usr/bin/env bash
# ==============================================================================
# Nabh-Drishti: Geospatial Calamity Prediction & Monitoring System
# NASA-IBM Prithvi-EO-100M Foundation Model + 6-Stage Visual Pipeline
# ==============================================================================

set -e

# Configure dynamic linker search paths and library locations for Tkinter / Tcl/Tk
export LD_LIBRARY_PATH="/home/bhuknu/Documents/Library_Books/SystemDesign/.lib/usr/lib:$LD_LIBRARY_PATH"
export TK_LIBRARY="/home/bhuknu/Documents/Library_Books/SystemDesign/.lib/usr/lib/tk8.6"
export TCL_LIBRARY="/usr/lib/tcl8.6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Starting Nabh-Drishti Calamity Prediction Pipeline..."
echo " Model: NASA-IBM Prithvi-EO-1.0-100M (Foundation Vision Transformer)"
echo " Environment: Python 3.14 + PyTorch + Pure Tkinter (Minimalist B&W)"
echo "======================================================================"

python3 src/app.py "$@"
