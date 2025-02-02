import numpy as np

from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from scipy.signal import convolve2d as conv2
from skimage.filters import median
from skimage.morphology import square


from ._metrics_utils import patch


def mean(F, B, c, f, b):
    name = 'MEAN'
    measure = np.nanmean(f)
    return name, measure


def rng(F, B, c, f, b):
    name = 'RNG'
    if len(f) > 0:
        measure = np.ptp(f)
    else:
        measure = np.nan  # Return NaN for empty arrays
    return name, measure


def var(F, B, c, f, b):
    name = 'VAR'
    measure = np.nanvar(f)
    return name, measure


def cv(F, B, c, f, b):
    name = 'CV'
    measure = (np.nanstd(f)/np.nanmean(f))*100
    return name, measure


def cpp(F, B, c, f, b):
    name = 'CPP'
    filt = np.array([[ -1/8, -1/8, -1/8],[-1/8, 1, -1/8],[ -1/8, -1/8,  -1/8]])
    I_hat = conv2(F, filt, mode='same')
    measure = np.nanmean(I_hat)
    return name, measure


def psnr(F, B, c, f, b):
    def _psnr(img1, img2):
        mse = np.square(np.subtract(img1, img2)).mean()
        return 20 * np.log10(np.nanmax(img1) / np.sqrt(mse))
    name = 'PSNR'
    I_hat = median(F/np.max(F), square(5))
    measure = _psnr(F, I_hat)
    return name, measure


def snr1(F, B, c, f, b):
    name = 'SNR1'
    bg_std = np.nanstd(b)
    measure = np.nanstd(f) / (bg_std + 1e-9)
    return name, measure


def snr2(F, B, c, f, b):
    name = 'SNR2'
    bg_std = np.nanstd(b)
    measure = np.nanmean(patch(F, 5)) / (bg_std + 1e-9)
    return name, measure 


def snr3(F, B, c, f, b):
    name = 'SNR3'
    fore_patch = patch(F, 5)
    std_diff = np.nanstd(fore_patch - np.nanmean(fore_patch))
    if std_diff == 0:
        std_diff = 1e-9
    measure = np.nanmean(fore_patch) / std_diff
    return name, measure


def snr4(F, B, c, f, b):
    name = 'SNR4'
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    bg_std = np.nanstd(back_patch)
    if bg_std == 0:
        bg_std = 1e-9
    measure = np.nanmean(fore_patch) / bg_std
    return name, measure


def snr5(F, B, c, f, b):
    name = 'SNR5'
    # Assume 'F' is the foreground image
    # Calculate local variance across the image using a sliding window approach
    window_size = 5  # Example window size
    local_variance = conv2(F**2, np.ones((window_size, window_size)), mode='valid') / window_size**2 - (conv2(F, np.ones((window_size, window_size)), mode='valid') / window_size)**2
    noise_estimate = np.sqrt(np.mean(local_variance))
    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)  # Adding a small number to avoid division by zero
    return name, measure


def snr6(F, B, c, f, b):
    name = 'SNR6'
    # Use Median Absolute Deviation (MAD) as a robust estimator of noise
    # MAD = median(|X_i - median(X)|)
    noise_estimate = np.median(np.abs(f - np.median(f))) / 0.6745  # 0.6745 is the consistency constant for normally distributed data
    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)  # Adding a small number to avoid division by zero
    return name, measure


def snr7(F, B, c, f, b):
    name = 'SNR7'
    # Use an edge detection filter (e.g., Sobel) to find edges
    edges = sobel(F)
    edge_pixels = F[(edges > np.percentile(edges, 95))]  # Consider top 5% of edges by magnitude
    noise_estimate = np.std(edge_pixels)
    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)
    return name, measure


def snr8(F, B, c, f, b):
    name = 'SNR8'
    # Transform the image to the frequency domain
    F_freq = np.fft.fft2(F)
    F_shifted = np.fft.fftshift(F_freq)
    # Assume noise is dominant in the outer regions; calculate standard deviation there
    rows, cols = F.shape
    crow, ccol = rows // 2 , cols // 2
    mask_size = 5  # Exclude the center
    mask = np.ones(F.shape, np.uint8)
    mask[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
    noise_freq = F_shifted * mask
    noise_time = np.fft.ifft2(np.fft.ifftshift(noise_freq)).real
    noise_estimate = np.std(noise_time)
    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)
    return name, measure


def snr9(F, B, c, f, b):
    name = 'SNR9'
    # Apply a texture filter (e.g., Local Binary Pattern) to identify texture variations
    LBP_texture = local_binary_pattern(F, P=8, R=1)  # Example parameters
    texture_regions = LBP_texture[(LBP_texture > np.percentile(LBP_texture, 95))]  # Consider top 5% of texture variance
    noise_estimate = np.std(texture_regions)
    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)
    return name, measure


def cnr(F, B, c, f, b):
    name = 'CNR'
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    measure = np.nanmean(fore_patch-back_patch) / (np.nanstd(back_patch) + 1e-6)
    return name, measure


def cvp(F, B, c, f, b):
    name = 'CVP'
    fore_patch = patch(F, 5)
    measure = np.nanstd(fore_patch) / (np.nanmean(fore_patch) + 1e-6 )
    return name, measure


def cjv(F, B, c, f, b):
    name = 'CJV'
    measure = (np.nanstd(f) + np.nanstd(b)) / abs(np.nanmean(f) - np.nanmean(b))
    return name, measure


def efc(F, B, c, f, b):
    name = 'EFC'
    n_vox = F.shape[0] * F.shape[1]
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
        np.log(1.0 / np.sqrt(n_vox))
    cc = (F**2).sum()
    b_max = np.sqrt(abs(cc))
    measure = float((1.0 / abs(efc_max)) * np.sum((F / b_max) * np.log((F + 1e16) / b_max)))
    return name, measure

def fber(F, B, c, f, b):
    name = 'FBER'
    fg_mu = np.nanmedian(np.abs(f) ** 2)
    bg_mu = np.nanmedian(np.abs(b) ** 2)
    if bg_mu < 1.0e-3:
        measure = 0
    measure = float(fg_mu / (bg_mu + 1e-6))
    return name, measure