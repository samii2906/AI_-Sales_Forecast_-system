PASTEL_COLORS = [
    "#b39ddb",  # lavender
    "#80cbc4",  # mint
    "#f48fb1",  # pink
    "#ffe082",  # yellow
    "#a5d6a7",  # green
    "#90caf9",  # blue
    "#ffcc80",  # peach
    "#ce93d8",  # purple
    "#ef9a9a",  # red
    "#80deea",  # cyan
]

CARD_STYLE = """
<style>
    /* Global dark base */
    .stApp { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #252840 100%);
        border: 1px solid rgba(179,157,219,0.25);
        border-radius: 14px;
        padding: 18px 22px;
        margin: 6px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(179,157,219,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #9e9e9e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-delta-pos { color: #a5d6a7; font-size: 0.8rem; font-weight: 600; }
    .metric-delta-neg { color: #f48fb1; font-size: 0.8rem; font-weight: 600; }

    .section-header {
        background: linear-gradient(90deg, rgba(179,157,219,0.15), transparent);
        border-left: 4px solid #b39ddb;
        padding: 12px 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 24px;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e0e0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9e9e9e !important;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #b39ddb !important;
        background: rgba(179,157,219,0.15) !important;
        font-weight: 700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #12151f !important;
        border-right: 1px solid rgba(179,157,219,0.1);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #b39ddb, #9575cd);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #9575cd, #7e57c2);
        box-shadow: 0 4px 15px rgba(149,117,205,0.4);
        transform: translateY(-1px);
    }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1d2e !important;
        border-radius: 8px !important;
        color: #b39ddb !important;
        font-weight: 600 !important;
    }

    /* Upload area */
    [data-testid="stFileUploadDropzone"] {
        background: #1a1d2e !important;
        border: 2px dashed rgba(179,157,219,0.4) !important;
        border-radius: 12px !important;
    }

    /* Alert boxes */
    .stSuccess { background: rgba(165,214,167,0.1) !important; border-color: #a5d6a7 !important; }
    .stWarning { background: rgba(255,224,130,0.1) !important; border-color: #ffe082 !important; }
    .stError   { background: rgba(244,143,177,0.1) !important; border-color: #f48fb1 !important; }
    .stInfo    { background: rgba(144,202,249,0.1) !important; border-color: #90caf9 !important; }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #b39ddb, #80cbc4) !important; }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] { background: #1a1d2e !important; }

    /* Chip/badge style */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-purple { background: rgba(179,157,219,0.2); color: #b39ddb; border: 1px solid #b39ddb55; }
    .badge-green  { background: rgba(165,214,167,0.2); color: #a5d6a7; border: 1px solid #a5d6a755; }
    .badge-red    { background: rgba(244,143,177,0.2); color: #f48fb1; border: 1px solid #f48fb155; }
    .badge-blue   { background: rgba(144,202,249,0.2); color: #90caf9; border: 1px solid #90caf955; }
</style>
"""

def inject_css():
    import streamlit as st
    st.markdown(CARD_STYLE, unsafe_allow_html=True)

def metric_card(label, value, delta=None, color="#b39ddb", icon=""):
    delta_html = ""
    if delta is not None:
        is_pos = "+" in str(delta) or (isinstance(delta, (int, float)) and delta >= 0)
        cls = "metric-delta-pos" if is_pos else "metric-delta-neg"
        arrow = "▲" if is_pos else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {delta_html}
    </div>
    """

def section_header(title, icon=""):
    import streamlit as st
    st.markdown(f'<div class="section-header"><h2>{icon} {title}</h2></div>', unsafe_allow_html=True)

def plotly_dark_layout():
    return dict(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d2e",
        font_color="#e0e0e0",
        font_family="sans-serif",
        title_font_size=15,
        title_font_color="#e0e0e0",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#333", borderwidth=1),
        xaxis=dict(gridcolor="#252840", linecolor="#333", tickcolor="#9e9e9e"),
        yaxis=dict(gridcolor="#252840", linecolor="#333", tickcolor="#9e9e9e"),
    )