from typing import Tuple, Union, Optional
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import fftpack
from nilearn import image, maskers
from numba import njit
from scipy.stats import rankdata
import time
import warnings

def compute_frequency_metrics(
    func_img: Union[str, Path, 'nib.nifti1.Nifti1Image'],
    mask_img: Optional[Union[str, Path, 'nib.nifti1.Nifti1Image']] = None,
    tr: float = 2.0,
    freq_range: Tuple[float, float] = (0.01, 0.08),
    standardize: bool = True
) -> Tuple['nib.nifti1.Nifti1Image', 'nib.nifti1.Nifti1Image']:
    """
    Calculate ALFF (Amplitude of Low Frequency Fluctuation) and fALFF (fractional ALFF).

    Parameters
    ----------
    func_img : str or Path or Nifti1Image
        Functional MRI data
    mask_img : str or Path or Nifti1Image, optional
        Brain mask. If None, uses whole-brain template
    tr : float, default=2.0
        Repetition time in seconds
    freq_range : tuple of float, default=(0.01, 0.08)
        Frequency range for ALFF calculation in Hz (low_freq, high_freq)
    standardize : bool, default=True
        Whether to standardize the time series before FFT

    Returns
    -------
    tuple
        (alff_img, falff_img): ALFF and fALFF NIfTI images
    """
    start_time = time.time()
    print('Computing ALFF and fALFF...')

    # Input validation
    if isinstance(func_img, (str, Path)):
        if not Path(func_img).exists():
            raise FileNotFoundError(f"Functional image not found: {func_img}")
        func_img = image.load_img(func_img)
    
    if mask_img is not None and isinstance(mask_img, (str, Path)):
        if not Path(mask_img).exists():
            raise FileNotFoundError(f"Mask image not found: {mask_img}")
        mask_img = image.load_img(mask_img)

    # Prepare mask
    if mask_img is None:
        masker = maskers.NiftiMasker(mask_strategy='whole-brain-template')
        masker.fit(func_img)
        mask_img = masker.mask_img_
    
    # Extract data
    func_data = image.clean_img(
        func_img,
        detrend=True,
        standardize='zscore' if standardize else False,
        t_r=tr
    ).get_fdata()
    mask_data = mask_img.get_fdata().astype(bool)

    if func_data.shape[:3] != mask_data.shape:
        raise ValueError("Functional and mask dimensions do not match.")

    # Prepare for FFT
    n_timepoints = func_data.shape[3]
    n_padded = int(2**np.ceil(np.log2(n_timepoints)))
    masked_data = func_data[mask_data]

    # Compute FFT
    fft_data = _compute_fft(masked_data, n_padded, tr)
    freq = fftpack.fftfreq(n_padded, d=tr)
    pos_freq_mask = freq > 0

    # Calculate frequency range indices
    low_freq, high_freq = freq_range
    freq_mask = (freq >= low_freq) & (freq <= high_freq) & pos_freq_mask
    
    # Calculate ALFF and fALFF
    alff_data, falff_data = _compute_alff_falff(fft_data[:, pos_freq_mask], freq_mask[pos_freq_mask])

    # Create output images
    alff_img, falff_img = _create_output_images(
        alff_data, falff_data, mask_data, func_img.affine, func_img.header
    )

    print(f'Computation completed in {time.time() - start_time:.2f} seconds.')
    return alff_img, falff_img

@njit
def _compute_fft(data: np.ndarray, n_padded: int, tr: float) -> np.ndarray:
    """Compute FFT with zero padding."""
    padded_data = np.pad(data, ((0, 0), (0, n_padded - data.shape[1])))
    return 2 * np.abs(fftpack.fft(padded_data, axis=1)) / data.shape[1]

@njit
def _compute_alff_falff(fft_data: np.ndarray, freq_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ALFF and fALFF from FFT data."""
    alff = np.mean(fft_data[:, freq_mask], axis=1)
    total_power = np.sum(fft_data, axis=1)
    falff = np.zeros_like(total_power)
    nonzero_mask = total_power > 0
    falff[nonzero_mask] = np.sum(fft_data[nonzero_mask][:, freq_mask], axis=1) / total_power[nonzero_mask]
    return alff, falff

def _create_output_images(
    alff_data: np.ndarray,
    falff_data: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    header: 'nib.nifti1.Nifti1Header'
) -> Tuple['nib.nifti1.Nifti1Image', 'nib.nifti1.Nifti1Image']:
    """Create NIfTI images from computed metrics."""
    alff_vol = np.zeros(mask.shape)
    falff_vol = np.zeros(mask.shape)
    alff_vol[mask] = alff_data
    falff_vol[mask] = falff_data
    return (
        nib.Nifti1Image(alff_vol, affine, header),
        nib.Nifti1Image(falff_vol, affine, header)
    )

def compute_reho(
    func_img: Union[str, Path, 'nib.nifti1.Nifti1Image'],
    mask_img: Optional[Union[str, Path, 'nib.nifti1.Nifti1Image']] = None,
    neighborhood_size: int = 27,
    n_jobs: int = -1
) -> 'nib.nifti1.Nifti1Image':
    """
    Calculate Regional Homogeneity (ReHo) map.

    Parameters
    ----------
    func_img : str or Path or Nifti1Image
        Functional MRI data
    mask_img : str or Path or Nifti1Image, optional
        Brain mask. If None, uses whole-brain template
    neighborhood_size : int, default=27
        Size of the neighborhood (7, 19, or 27)
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means using all processors

    Returns
    -------
    nib.nifti1.Nifti1Image
        ReHo map as a NIfTI image
    """
    if neighborhood_size not in {7, 19, 27}:
        raise ValueError("neighborhood_size must be 7, 19, or 27")

    start_time = time.time()
    print('Computing ReHo...')

    # Load and validate data
    func_img = image.load_img(func_img) if isinstance(func_img, (str, Path)) else func_img
    
    if mask_img is None:
        masker = maskers.NiftiMasker(mask_strategy='whole-brain-template')
        mask_img = masker.fit(func_img).mask_img_
    else:
        mask_img = image.load_img(mask_img) if isinstance(mask_img, (str, Path)) else mask_img

    # Compute ranks
    func_data = func_img.get_fdata()
    mask_data = mask_img.get_fdata().astype(bool)
    
    if func_data.shape[:3] != mask_data.shape:
        raise ValueError("Functional and mask dimensions do not match")

    # Get ranks and compute ReHo
    global_ranks = _compute_ranks(func_data, mask_data)
    reho_map = _compute_reho_map(global_ranks, mask_data, neighborhood_size, n_jobs)

    # Create output image
    reho_img = nib.Nifti1Image(reho_map, mask_img.affine, mask_img.header)
    
    print(f'Computation completed in {time.time() - start_time:.2f} seconds.')
    return reho_img

@njit(parallel=True)
def _compute_ranks(func_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute ranks of time series within mask."""
    masked_data = func_data[mask]
    ranks = np.zeros_like(masked_data, dtype=np.float64)
    for i in range(masked_data.shape[0]):
        ranks[i] = rankdata(masked_data[i], method='average')
    return ranks

@njit
def _get_neighborhood_indices(
    x: int, y: int, z: int,
    mask: np.ndarray,
    neighborhood_size: int
) -> np.ndarray:
    """Get indices of neighboring voxels within mask."""
    if neighborhood_size == 7:
        offsets = np.array([(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)])
    elif neighborhood_size == 19:
        offsets = np.array([
            (x,y,z) for x in range(-1,2) for y in range(-1,2) for z in range(-1,2)
            if abs(x) + abs(y) + abs(z) <= 2
        ])
    else:  # 27
        offsets = np.array([
            (x,y,z) for x in range(-1,2) for y in range(-1,2) for z in range(-1,2)
        ])

    indices = []
    for dx, dy, dz in offsets:
        nx, ny, nz = x + dx, y + dy, z + dz
        if (0 <= nx < mask.shape[0] and 
            0 <= ny < mask.shape[1] and 
            0 <= nz < mask.shape[2] and 
            mask[nx, ny, nz]):
            indices.append((nx, ny, nz))
    return np.array(indices)

def _compute_reho_map(
    ranks: np.ndarray,
    mask: np.ndarray,
    neighborhood_size: int,
    n_jobs: int
) -> np.ndarray:
    """Compute ReHo map using parallel processing."""
    reho_map = np.zeros(mask.shape, dtype=np.float64)
    coords = np.array(np.where(mask)).T

    from joblib import Parallel, delayed
    
    def process_voxel(coord):
        x, y, z = coord
        neighbors = _get_neighborhood_indices(x, y, z, mask, neighborhood_size)
        if len(neighbors) < 2:
            return 0.0
        
        neighbor_ranks = ranks[neighbors]
        n = neighbor_ranks.shape[1]  # timepoints
        k = len(neighbors)  # neighbors
        
        # Compute KCC
        rank_sum = np.sum(neighbor_ranks, axis=0)
        rank_mean = np.mean(rank_sum)
        S = np.sum((rank_sum - rank_mean) ** 2)
        return 12 * S / (k ** 2 * (n ** 3 - n))

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_voxel)(coord) for coord in coords
    )
    
    for coord, value in zip(coords, results):
        reho_map[tuple(coord)] = value

    return reho_map