from nilearn import maskers, image
import numpy as np
import time
from pathlib import Path
from typing import Union, Optional
import warnings

def compute_seed_to_voxel_fc(
    seed_img: Union[str, Path, 'image.Nifti1Image'],
    func_img: Union[str, Path, 'image.Nifti1Image'],
    brain_mask_img: Optional[Union[str, Path, 'image.Nifti1Image']] = None,
    standardize: str = 'zscore_sample',
    memory_efficient: bool = True,
    n_jobs: int = -1
) -> 'image.Nifti1Image':
    """
    Compute seed-to-voxel functional connectivity (FC) maps.

    Parameters
    ----------
    seed_img : str or Path or NiftiImage
        Path to NIfTI file of the ROI mask or a Nifti object
    func_img : str or Path or NiftiImage
        Path to NIfTI file of the functional MRI data or a Nifti object
    brain_mask_img : str or Path or NiftiImage, optional
        Path to NIfTI file of the brain mask or a Nifti object
    standardize : str, default='zscore_sample'
        Standardization strategy, options: {'zscore_sample', 'scale', False}
    memory_efficient : bool, default=True
        Whether to use memory efficient mode
    n_jobs : int, default=-1
        Number of parallel processes, -1 means using all available CPUs

    Returns
    -------
    image.Nifti1Image
        Seed-to-voxel correlation map as a NIfTI image

    Raises
    ------
    ValueError
        When input parameters are invalid
    FileNotFoundError
        When input files do not exist
    """
    
    # Parameter validation
    if isinstance(seed_img, (str, Path)):
        if not Path(seed_img).exists():
            raise FileNotFoundError(f"Seed mask file not found: {seed_img}")
    
    if isinstance(func_img, (str, Path)):
        if not Path(func_img).exists():
            raise FileNotFoundError(f"Functional image file not found: {func_img}")
    
    if brain_mask_img is not None and isinstance(brain_mask_img, (str, Path)):
        if not Path(brain_mask_img).exists():
            raise FileNotFoundError(f"Brain mask file not found: {brain_mask_img}")

    time_start = time.time()
    print('Computing seed-to-voxel functional connectivity...')

    try:
        # Extract ROI time series
        roi_masker = maskers.NiftiLabelsMasker(
            seed_img,
            standardize=standardize,
            memory_level=1 if memory_efficient else 0,
            n_jobs=n_jobs
        )
        roi_time_series = roi_masker.fit_transform(func_img)

        # Configure brain mask
        if brain_mask_img is None:
            brain_masker = maskers.NiftiMasker(
                standardize=standardize,
                mask_strategy='whole-brain-template',
                memory_level=1 if memory_efficient else 0,
                n_jobs=n_jobs
            )
        else:
            brain_masker = maskers.NiftiMasker(
                brain_mask_img,
                standardize=standardize,
                memory_level=1 if memory_efficient else 0,
                n_jobs=n_jobs
            )
        
        # Extract brain time series
        brain_time_series = brain_masker.fit_transform(func_img)

        # Compute correlations
        # Use optimized numpy operations for correlation calculation
        n_samples = roi_time_series.shape[0]
        seed_to_voxel_correlations = np.dot(brain_time_series.T, roi_time_series) / n_samples

        # Transform correlation values back to NIfTI image space
        seed_to_voxel_correlations_img = brain_masker.inverse_transform(
            seed_to_voxel_correlations.T
        )
        
        time_end = time.time()
        print(f'Functional connectivity computation completed in {round(time_end - time_start, 2)} seconds.')
        
        return seed_to_voxel_correlations_img

    except Exception as e:
        print(f'Error occurred during computation: {str(e)}')
        raise
