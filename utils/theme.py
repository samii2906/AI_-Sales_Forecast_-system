PASTEL_COLORS = [
    "#b39ddb",  # lavender
    "#80cbc4",  # mint
    "#f48fb1",  # pink
    "#ffe082",  # yellow
    "#a5d6a7",  # green
    "#90caf9",  # blue
    "#ffcc80",  # peach
    "#ce93d8",  # purple
]

CARD_STYLE = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #b39ddb33;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(179,157,219,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #b39ddb;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9e9e9e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-header {
        background: linear-gradient(90deg, #b39ddb22, transparent);
        border-left: 3px solid #b39ddb;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9e9e9e;
    }
    .stTabs [aria-selected="true"] {
        color: #b39ddb !important;
        border-bottom: 2px solid #b39ddb !important;
    }
</style>
"""

def inject_css():
    import streamlit as st
    st.markdown(CARD_STYLE, unsafe_allow_html=True)

def metric_card(label, value, delta=None, color="#b39ddb"):
    delta_html = f'<span style="color: {"#a5d6a7" if "+" in str(delta) else "#f48fb1"}; font-size:0.8rem">{delta}</span>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {delta_html}
    </div>
    """