import numpy as np
from scipy.optimize import minimize
from scipy.stats import powerlaw, mle

def marchenko_pastur_lambda_plus(n, T):
    """
    Upper bound of Marchenko‑Pastur distribution for random matrix.
    n: number of assets
    T: number of observations (window)
    """
    sigma = 1.0  # we assume standardised returns
    q = n / T
    lambda_plus = sigma ** 2 * (1 + np.sqrt(q)) ** 2
    return lambda_plus

def power_law_mle(data):
    """
    Fit power law p(x) ~ x^{-gamma} for x >= x_min.
    Returns gamma, x_min.
    """
    # Use built‑in powerlaw.fit, but we need to estimate x_min.
    from powerlaw import Fit
    fit = Fit(data, discrete=False, verbose=False)
    gamma = fit.power_law.alpha
    x_min = fit.power_law.xmin
    return gamma, x_min

def compute_criticality(returns_df, window):
    """
    Compute criticality metrics for a given window.
    Returns:
        gamma: power‑law exponent of eigenvalue tail
        lambda_max: largest eigenvalue
        mp_upper: Marchenko‑Pastur upper bound
        participation_ratio: vector of PR for each ETF (eigenvector localization)
        top_etfs: list of ETFs with highest PR
    """
    if len(returns_df) < window:
        return None
    # Standardise returns
    data = returns_df.iloc[-window:].dropna(axis=1, how='any')
    if data.shape[1] < 5:
        return None
    # Compute correlation matrix
    corr = data.corr().values
    # Eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = eigvals[::-1]  # descending
    eigvecs = eigvecs[:, ::-1]
    # Marchenko‑Pastur upper bound
    n = eigvals.shape[0]
    T = window
    lambda_mp = marchenko_pastur_lambda_plus(n, T)
    # Select eigenvalues > lambda_mp for power‑law fitting
    tail_eigs = eigvals[eigvals > lambda_mp]
    if len(tail_eigs) < config.MIN_EIGENVALUES:
        gamma = np.nan
    else:
        gamma, _ = power_law_mle(tail_eigs)
    # Participation ratio (PR) for the largest eigenvalue
    v = eigvecs[:, 0]
    pr = np.sum(v ** 4)   # lower PR = more delocalized, higher PR = localized
    # Actually we need per‑ETF participation: p_i = v_i^4 / sum_j v_j^4? For ranking we use squared eigenvector components.
    # Influence of each ETF: the component of the largest eigenvector squared.
    influence = v ** 2
    # Top ETFs by influence
    etf_names = data.columns.tolist()
    top_idx = np.argsort(influence)[::-1][:config.TOP_N]
    top_etfs = [{"ticker": etf_names[i], "influence": float(influence[i])} for i in top_idx]
    # Criticality score: distance from critical range (if gamma is in range, score low (near critical), else high)
    if not np.isnan(gamma):
        if config.GAMMA_CRITICAL_LOW <= gamma <= config.GAMMA_CRITICAL_HIGH:
            distance_to_critical = 0.0
            cash_allocation = 0.2   # increase cash to 20%
        elif gamma < config.GAMMA_CRITICAL_LOW:
            distance_to_critical = config.GAMMA_CRITICAL_LOW - gamma
            cash_allocation = 0.05
        else:
            distance_to_critical = gamma - config.GAMMA_CRITICAL_HIGH
            cash_allocation = 0.05
    else:
        distance_to_critical = np.nan
        cash_allocation = 0.0

    return {
        "gamma": float(gamma) if not np.isnan(gamma) else None,
        "lambda_max": float(eigvals[0]),
        "lambda_mp": float(lambda_mp),
        "distance_to_critical": float(distance_to_critical) if not np.isnan(distance_to_critical) else None,
        "cash_allocation": float(cash_allocation),
        "top_etfs": top_etfs,
        "participation_ratio": float(pr)
    }
