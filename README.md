# Neuronal Avalanche Criticality Engine

Detects self‑organised criticality in ETF correlation spectra.  
Fits a power‑law to the eigenvalue tail (above Marchenko‑Pastur bound) and computes exponent γ.  
Near‑critical markets (γ ∈ [1.5,2.5]) increase cash allocation.

- **Rolling windows:** 60, 126, 252 days
- **Output per universe:**
  - Top 3 ETFs by participation in the largest eigenvector (influence)
  - Recommended cash allocation
  - γ and eigenvalue metrics for each window
- Runs daily on GitHub Actions

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
