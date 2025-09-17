<p align="center">
  <img src="assets/icons/NeuroNavigatAR_logo.png" alt="NNAR Logo" width="300"/>
</p>

# NeuroNavigatAR (NNAR) - Augmented Reality Tool for Real-Time Optode/Electrode Placement

NeuroNavigatAR (NNAR) is an augmented reality (AR) tool designed to visualize optode/electrode positions based on the 10-20 (10-10, 10-5) system in real time. This tool assists users locating the sensor positions for neuroimaging setups.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Features
- Real-time AR visualization of optode/electrode positions (10-20, 10-10, 10-5 systems)
- Supports Colin27 and age-matched atlases (ages 20–84 in 5-year intervals)
- Works with external webcams or built-in laptop cameras

## Installation

### Option 1: Install from GitHub (Recommended)
```bash
pip install git+https://github.com/COTILab/NeuroNavigatAR.git
```

### Option 2: Install from source
```bash
git clone https://github.com/COTILab/NeuroNavigatAR.git
cd NeuroNavigatAR
pip install .
```

## Usage
### GUI application
After installation, you can run the application in several ways:
#### Option 1: Command line entry point
```bash
neuronavigatar
```

#### Option 2: Python script
```bash
from nnar import nnar
import sys
from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)
window = nnar()
window.show()
sys.exit(app.exec_())
```

## Legacy Usage (for existing users)
If you have the old setup, you can still use:
```bash
# Using Conda environment (if you have environment.yml)
conda env create -f environment.yml
conda activate nnar
python main.py
```

## Citation
If you use this tool in your research, please cite the following papers:

F. Yen, Y. Lin, and Q. Fang, "Real-time guidance for fNIRS headgear placement using augmented reality," in Optica Biophotonics Congress: Biomedical Optics 2024 (Translational, Microscopy, OCT, OTS, BRAIN), Technical Digest Series (Optica Publishing Group, 2024), paper BW1C.6.

Dai, H., Pears, N., Smith, W., & Duncan, C. (2019). Statistical Modeling of Craniofacial Shape and Texture. International Journal of Computer Vision, 128(2), 547–571. https://doi.org/10.1007/s11263-019-01260-7
