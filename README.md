# RestPy

A Python package for resting-state fMRI analysis.

## Features

- Seed-based functional connectivity analysis
- Spherical ROI creation
- Brain metrics calculation (ALFF, fALFF, ReHo)

## Installation
Clone the repository
```bash
git clone https://github.com/yourusername/restpy.git
```
Navigate to the directory
```bash
cd restpy
```
Install the package
```bash
pip install -e .
```
## Usage
```python
from restpy import (
compute_seed_based_connectivity,
create_spherical_roi,
compute_alff,
compute_falff,
compute_reho
)
```
Create ROI
```python
roi_masker, roi_img = create_spherical_roi(
coords=(-45, 0, 56), # MNI coordinates
radius=8,
target_img="func.nii.gz"
)
```
Compute functional connectivity
```python
fc_map = compute_seed_based_connectivity(
seed_img="seed.nii.gz",
func_img="func.nii.gz"
)
```
Calculate brain metrics
```python
alff_img = compute_alff("func.nii.gz")
falff_img = compute_falff("func.nii.gz")
reho_img = compute_reho("func.nii.gz")
```
