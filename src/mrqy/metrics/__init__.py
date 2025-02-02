from ._metrics import (
    mean,
    rng,
    var,
    cv,
    cpp,
    psnr,
    snr1,
    snr2,
    snr3,
    snr4,
    snr5,
    snr6,
    snr7,
    snr8,
    snr9,
    cnr,
    cvp,
    cjv,
    efc,
    fber
)

MFUNCS = [
    "mean",
    "rng",
    "var",
    "cv",
    "cpp",
    "psnr",
    "snr1",
    "snr2",
    "snr3",
    "snr4",
    "snr5",
    "snr6",
    "snr7",
    "snr8",
    "snr9",
    "cnr",
    "cvp",
    "cjv",
    "efc",
    "fber"
]


__all__ = [
    "MFUNCS",
    *MFUNCS
]
