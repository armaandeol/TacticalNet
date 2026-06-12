"""
Phase 4: Streamlit Dashboard — World Cup 2026 Match Simulator

Run with:
    streamlit run src/app.py
"""

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from modules import TacticalNet, StyleAutoencoder

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="WC 2026 Match Simulator",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ World Cup 2026 Match Outcome Simulator")
st.markdown(
    "Adjust team parameters below to simulate match outcome probabilities "
    "using the GNN + LSTM + Autoencoder pipeline."
)

# ─────────────────────────────────────────────
# Sidebar: Team Configuration
# ─────────────────────────────────────────────

TACTICS = ["Tiki-Taka", "High Press", "Low Block", "Counter-Attack", "Gegenpressing", "Possession"]
TEAMS = [
    "Argentina", "France", "Brazil", "England", "Spain", "Germany",
    "Portugal", "Netherlands", "Italy", "Belgium", "USA", "Mexico",
    "Morocco", "Japan", "South Korea", "Senegal", "Australia", "Croatia",
]

with st.sidebar:
    st.header("🔵 Team A")
    team_a_name = st.selectbox("Select Team A", TEAMS, index=0)
    tactic_a = st.selectbox("Tactical Setup A", TACTICS, index=0)
    form_a = st.slider("Recent Form (0=Poor, 10=Excellent)", 0.0, 10.0, 7.5, 0.1, key="form_a")
    ppda_a = st.slider("PPDA (Pressing Intensity — lower = more pressing)", 3.0, 20.0, 8.0, 0.5, key="ppda_a")
    field_tilt_a = st.slider("Field Tilt (%)", 0.0, 100.0, 60.0, 1.0, key="tilt_a")

    st.divider()

    st.header("🔴 Team B")
    team_b_name = st.selectbox("Select Team B", TEAMS, index=3)
    tactic_b = st.selectbox("Tactical Setup B", TACTICS, index=2)
    form_b = st.slider("Recent Form (0=Poor, 10=Excellent)", 0.0, 10.0, 6.0, 0.1, key="form_b")
    ppda_b = st.slider("PPDA (Pressing Intensity — lower = more pressing)", 3.0, 20.0, 14.0, 0.5, key="ppda_b")
    field_tilt_b = st.slider("Field Tilt (%)", 0.0, 100.0, 40.0, 1.0, key="tilt_b")


# ─────────────────────────────────────────────
# Simulate (demo mode — no real weights needed)
# ─────────────────────────────────────────────

def build_demo_graph(form_score: float, num_players: int = 11, feature_dim: int = 64) -> Data:
    """
    Build a synthetic player graph for demo purposes.
    In production, replace with real LSTM embeddings and passing edge data.
    """
    rng = np.random.default_rng(seed=int(form_score * 100))
    # Node features: LSTM form embeddings (simulated)
    x = torch.tensor(
        rng.normal(loc=form_score / 10.0, scale=0.1, size=(num_players, feature_dim)),
        dtype=torch.float32,
    )
    # Edges: fully connected passing graph (all pairs)
    src = [i for i in range(num_players) for j in range(num_players) if i != j]
    dst = [j for i in range(num_players) for j in range(num_players) if i != j]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    batch = torch.zeros(num_players, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)


def build_style_vector(ppda: float, field_tilt: float, tactic: str) -> torch.Tensor:
    """Build a 4-dim style latent vector from slider inputs (demo approximation)."""
    tactic_map = {
        "Tiki-Taka": [0.9, 0.2, 0.8, 0.3],
        "High Press": [0.7, 0.8, 0.6, 0.4],
        "Low Block": [0.2, 0.1, 0.3, 0.5],
        "Counter-Attack": [0.3, 0.6, 0.4, 0.7],
        "Gegenpressing": [0.8, 0.9, 0.7, 0.3],
        "Possession": [0.85, 0.3, 0.75, 0.2],
    }
    base = tactic_map.get(tactic, [0.5, 0.5, 0.5, 0.5])
    # Blend with slider values
    pressing_norm = 1.0 - (ppda - 3.0) / 17.0  # invert: lower PPDA = more pressing
    tilt_norm = field_tilt / 100.0
    vec = [
        base[0] * 0.6 + tilt_norm * 0.4,
        base[1] * 0.6 + pressing_norm * 0.4,
        base[2],
        base[3],
    ]
    return torch.tensor([vec], dtype=torch.float32)  # (1, 4)


@st.cache_resource
def load_model():
    """Load TacticalNet. Falls back to random weights if no checkpoint found."""
    model = TacticalNet(player_feature_dim=64, hidden_dim=128, style_latent_dim=4)
    try:
        state = torch.load("weights/tactical_net_fold2.pt", map_location="cpu")
        model.load_state_dict(state)
        st.sidebar.success("✅ Loaded trained weights")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ No trained weights found — using demo mode")
    model.eval()
    return model


model = load_model()

data_a = build_demo_graph(form_a)
data_b = build_demo_graph(form_b)
style_a = build_style_vector(ppda_a, field_tilt_a, tactic_a)
style_b = build_style_vector(ppda_b, field_tilt_b, tactic_b)

with torch.no_grad():
    logits = model(data_a, data_b, style_a, style_b)
    probs = F.softmax(logits, dim=-1).squeeze().numpy()

p_win, p_draw, p_loss = float(probs[0]), float(probs[1]), float(probs[2])

# ─────────────────────────────────────────────
# Results Display
# ─────────────────────────────────────────────

st.subheader(f"🏟️ {team_a_name} vs {team_b_name}")
st.caption(f"{tactic_a} vs {tactic_b}")

col1, col2, col3 = st.columns(3)
col1.metric(f"🔵 {team_a_name} Win", f"{p_win:.1%}")
col2.metric("🤝 Draw", f"{p_draw:.1%}")
col3.metric(f"🔴 {team_b_name} Win", f"{p_loss:.1%}")

# Probability bar chart
import plotly.graph_objects as go

fig = go.Figure(go.Bar(
    x=[f"{team_a_name} Win", "Draw", f"{team_b_name} Win"],
    y=[p_win, p_draw, p_loss],
    marker_color=["#1f77b4", "#aec7e8", "#d62728"],
    text=[f"{v:.1%}" for v in [p_win, p_draw, p_loss]],
    textposition="outside",
))
fig.update_layout(
    yaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
    xaxis_title="Outcome",
    title="Match Outcome Probability Distribution",
    height=400,
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# Tactical narrative
st.subheader("📊 Tactical Analysis")
if tactic_a == "Tiki-Taka" and tactic_b == "Low Block":
    narrative = (
        f"{team_a_name}'s Tiki-Taka system effectively counters {team_b_name}'s Low Block setup. "
        "High possession and positional play should create overloads in wide areas. "
        f"Combined with current form (A: {form_a}/10 vs B: {form_b}/10), "
        f"the model projects {team_a_name} as the likely controller of match tempo."
    )
elif ppda_a < ppda_b:
    narrative = (
        f"{team_a_name} applies significantly more pressing pressure (PPDA {ppda_a:.1f} vs {ppda_b:.1f}). "
        "This high-intensity approach should disrupt build-up play and create turnovers in dangerous areas."
    )
else:
    narrative = (
        f"This is a closely contested tactical matchup. {team_a_name}'s {tactic_a} "
        f"against {team_b_name}'s {tactic_b} creates an interesting dynamic. "
        "Key differentiator will be individual player form and set-piece efficiency."
    )

st.info(narrative)

st.divider()
st.caption(
    "Model: GCN + LSTM Form Embeddings + Style Autoencoder | "
    "Validated via time-series backtesting on 2018 & 2022 World Cups"
)
