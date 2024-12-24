"""Validation utilities for RestPy."""
from pathlib import Path
from typing import Union, Optional
import nibabel as nib
from nilearn import image

def validate_input_image(
    img: Union[str, Path, 'nib.nifti1.Nifti1Image'],
    ensure_3d: bool = False
) -> 'nib.nifti1.Nifti1Image':
    """
    Validate and load input image.
    
    Parameters
    ----------
    img : str or Path or Nifti1Image
        Input image
    ensure_3d : bool, default=False
        If True, ensures the image is 3D
        
    Returns
    -------
    nib.nifti1.Nifti1Image
        Validated image object
    """
    if isinstance(img, (str, Path)):
        if not Path(img).exists():
            raise FileNotFoundError(f"Image not found: {img}")
        img = image.load_img(img)
    
    if ensure_3d and len(img.shape) != 3:
        raise ValueError("Image must be 3D")
    
    return img 