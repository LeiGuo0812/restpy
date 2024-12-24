from nilearn.maskers import NiftiSpheresMasker, NiftiMasker
from nilearn import plotting, image
from nilearn import datasets
import numpy as np
from typing import Tuple, Union, Optional
from pathlib import Path

def create_sphere_mask(
    coords: Tuple[float, float, float],
    radius: float,
    target_img: Union[str, Path, 'image.Nifti1Image'],
    constraint_mask: Optional[Union[str, Path, 'image.Nifti1Image']] = None,
    plot_results: bool = True
) -> Tuple[NiftiSpheresMasker, 'image.Nifti1Image']:
    """
    Create a spherical mask at specified coordinates, constrained by an optional brain mask
    and resampled to match the target image space.

    Parameters
    ----------
    coords : tuple
        (x, y, z) coordinates for the sphere center in MNI space
    radius : float
        Radius of the sphere in millimeters
    target_img : str or Path or NiftiImage
        Target image to which the sphere mask will be resampled. This is typically
        your functional or anatomical image that will be used in subsequent analyses
    constraint_mask : str or Path or NiftiImage, optional
        Brain mask used to constrain the sphere mask (e.g., gray matter mask or whole brain mask).
        If None, uses nilearn's default whole-brain template
    plot_results : bool, default=True
        Whether to display visualization plots

    Returns
    -------
    tuple
        (sphere_masker, mask_img) where:
        - sphere_masker: fitted NiftiSpheresMasker object ready for use
        - mask_img: binary mask image in the same space as target_img
    """
    # Validate inputs
    if isinstance(target_img, (str, Path)):
        if not Path(target_img).exists():
            raise FileNotFoundError(f"Target image not found: {target_img}")
    
    if constraint_mask is not None and isinstance(constraint_mask, (str, Path)):
        if not Path(constraint_mask).exists():
            raise FileNotFoundError(f"Constraint mask not found: {constraint_mask}")

    # Create constraint mask if not provided
    if constraint_mask is None:
        print("No constraint mask provided. Using default whole-brain template.")
        brain_masker = NiftiMasker(mask_strategy='whole-brain-template')
        brain_masker.fit(target_img)
        constraint_mask = brain_masker.mask_img_
    
    if plot_results:
        # Plot the target image
        plotting.plot_epi(image.index_img(target_img, 0),
                         title="Target Image (First Volume)")
        # Plot the constraint mask
        plotting.plot_glass_brain(constraint_mask,
                                title="Constraint Mask")

    # Create sphere masker with constraint mask
    sphere_masker = NiftiSpheresMasker(
        seeds=[coords],
        radius=radius,
        mask_img=constraint_mask,
        verbose=1
    )

    # Fit the masker to the target image to ensure proper resampling
    sphere_masker.fit(target_img)
    
    # Create the binary sphere mask image in target space
    mask_img = sphere_masker.inverse_transform(np.ones(1))
    
    if plot_results:
        # Plot the final sphere mask
        plotting.plot_glass_brain(mask_img,
                                title=f"Sphere Mask (radius={radius}mm)")
        # Plot the overlay of mask on target image
        plotting.plot_roi(mask_img, target_img,
                         title="Sphere Mask Overlay on Target Image")

    return sphere_masker, mask_img
