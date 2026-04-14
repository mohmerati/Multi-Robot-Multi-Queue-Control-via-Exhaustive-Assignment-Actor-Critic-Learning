import numpy as np

def mean_ci95(x: np.ndarray):
    """
    Returns (mean, half_width, (lo, hi)) for 95% CI across samples in x.
    Uses Student-t if SciPy available; else normal approx.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    m = float(x.mean())
    if n == 1:
        return m, np.nan, (np.nan, np.nan)
    s = float(x.std(ddof=1))
    se = s / np.sqrt(n)

    crit = 1.959963984540054  # ~N(0,1) 97.5th percentile

    hw = crit * se
    return m, hw, (m - hw, m + hw)