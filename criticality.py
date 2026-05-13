import numpy as np
import config

def marchenko_pastur_lambda_plus(n, T):
    """Upper bound of Marchenko‑Pastur distribution."""
    sigma = 1.0
    q = n / T
    lambda_plus = sigma ** 2 * (1 + np.sqrt(q)) ** 2
    return lambda_plus

def power_law_mle(data):
    """
    MLE for power‑law exponent gamma.
    Assumes data >= x_min, where x_min = min(data).
    Returns gamma, x_min.
    """
    x_min = np.min(data)
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    log_ratios = np.log(data / x_min)
    gamma = 1 + n / np.sum(log_ratios)
    return gamma, x_min

def compute_criticality(returns_df, window):
    """
    Compute criticality metrics for a given window.
    """
    if len(returns_df) < window:
        return None
    data = returns_df.iloc[-window:].dropna(axis=1, how='any')
    if data.shape[1] < 5:
        return None

    corr = data.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = eigvals[::-1]               # descending
    eigvecs = eigvecs[:, ::-1]

    n = eigvals.shape[0]
    T = window
    lambda_mp = marchenko_pastur_lambda_plus(n, T)

    # eigenvalues larger than Marchenko‑Pastur upper bound
    tail_eigs = eigvals[eigvals > lambda_mp]
    if len(tail_eigs) < config.MIN_EIGENVALUES:
        gamma = np.nan
    else:
        gamma, _ = power_law_mle(tail_eigs)

    # participation of each ETF in the largest eigenvector
    v = eigvecs[:, 0]
    influence = v ** 2
    etf_names = data.columns.tolist()
    top_idx = np.argsort(influence)[::-1][:config.TOP_N]
    top_etfs = [{"ticker": etf_names[i], "influence": float(influence[i])} for i in top_idx]

    # cash allocation based on criticality
    if not np.isnan(gamma):
        if config.GAMMA_CRITICAL_LOW <= gamma <= config.GAMMA_CRITICAL_HIGH:
            distance_to_critical = 0.0
            cash_allocation = 0.20   # 20% cash near critical
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
        "participation_ratio": float(np.sum(v ** 4))
    }
