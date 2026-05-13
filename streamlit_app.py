import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Neuronal Avalanche Criticality", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Neuronal Avalanche Criticality Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Self‑organised criticality via eigenvalue power‑law tails | Cash allocation signal</div>', unsafe_allow_html=True)

st.sidebar.markdown("## ⚡ Criticality")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** Power‑law fit of eigenvalue tail vs. Marchenko‑Pastur")
st.sidebar.markdown(f"**Critical range:** {config.GAMMA_CRITICAL_LOW}–{config.GAMMA_CRITICAL_HIGH}")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'criticality_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Influence (Critical Mode)")
st.markdown("*ETFs with highest participation in the largest eigenvector – likely drivers of criticality.*")

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">influence = {etf['influence']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    # Show cash allocation
    cash = uni_data.get("cash_allocation", 0.0)
    st.info(f"💵 Recommended cash allocation: **{cash:.0%}** of portfolio (criticality signal)")
    # Expandable criticality metrics per window
    with st.expander("📊 Criticality metrics per window"):
        crit = uni_data.get("criticality_results", {})
        if crit:
            rows = []
            for win_key, vals in crit.items():
                win = win_key.split("_")[1]
                rows.append({
                    "Window (days)": win,
                    "γ (power‑law exponent)": vals.get("gamma"),
                    "λ_max / λ_MP": f"{vals.get('lambda_max',0):.2f} / {vals.get('lambda_mp',0):.2f}",
                    "Distance to critical": vals.get("distance_to_critical", 0),
                    "Cash allocation": f"{vals.get('cash_allocation',0):.0%}"
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("γ between 1.5 and 2.5 indicates criticality (max susceptibility). Cash allocation increases near criticality to reduce portfolio risk.")
