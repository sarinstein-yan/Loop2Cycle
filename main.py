# main.py
import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from src import (
    DualCycleAnalyzer, 
    render_theory_intro, 
    section_conceptual_diagrams
)

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "assets"
HAM_PATH = DATA_DIR / "Ham_96.csv"
CACHE_PATH = DATA_DIR / "nmi_cache_k3.pkl"
N_CLUSTERS_FIXED = 3  # KMeans k is fixed to 3 throughout the app.
MAX_CYCLE_LENGTH = 88  # Precompute MI cycles from default length up to this cap.



# ---------------------------------------------------------------------
# UI helper: loop-dependent defaults
# ---------------------------------------------------------------------
def _get_ui_defaults_for_loop(
    loop_label: int, labels_full: np.ndarray
) -> Tuple[int, int]:
    """Return (cycle_length, mi_rank) defaults for a given spectral loop label.

    Hard-coded special cases:
      - loop label 0 -> L = 88, NMI rank = 1
      - loop label 2 -> L = 32, NMI rank = 2

    For any other label, fall back to using the loop size (number of
    eigenvalues in the cluster) as the cycle length and NMI rank = 1.
    """
    if loop_label == 0:
        return 88, 1
    if loop_label == 2:
        return 32, 2

    cluster_size = int((labels_full == loop_label).sum())
    cycle_length = int(min(max(cluster_size, 1), MAX_CYCLE_LENGTH))
    return cycle_length, 1


# ---------------------------------------------------------------------
# Helper: load / build analyzer + disk cache
# ---------------------------------------------------------------------
def _load_or_build_nmi_cache(analyzer: DualCycleAnalyzer) -> None:
    """Attach precomputed NMI cache to `analyzer`.

    If a compatible cache exists (same Hamiltonian digest & k), it is loaded.
    Otherwise, NMI cycles are computed once for k=3 and written to disk.

    The smallest-count cluster (isolated eigenvalues) is ignored in the cache.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Digest of Hamiltonian
    ham_digest = hashlib.sha256(analyzer.H.tobytes()).hexdigest()

    # Try to load existing cache ------------------------------------------------
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                payload = pickle.load(f)

            if (
                isinstance(payload, dict)
                and payload.get("ham_digest") == ham_digest
                and int(payload.get("n_clusters", -1)) == N_CLUSTERS_FIXED
                and "labels" in payload
                and "mi_cache" in payload
            ):
                analyzer.labels = np.array(payload["labels"])
                analyzer.n_clusters = N_CLUSTERS_FIXED
                analyzer._mi_cache = payload["mi_cache"]

                ignored = payload.get("ignored_cluster")
                if ignored is None:
                    uniq, counts = np.unique(analyzer.labels, return_counts=True)
                    ignored = int(uniq[np.argmin(counts)])

                analyzer._ignored_cluster_label = int(ignored)
                analyzer._max_cycle_length = int(
                    payload.get("max_cycle_length", MAX_CYCLE_LENGTH)
                )
                return
        except Exception as e:
            print(f"Warning: failed to load cache at {CACHE_PATH}: {e}")

    # --------------------------------------------------------------------------
    # No valid cache -> compute once (sequentially) and save
    # --------------------------------------------------------------------------
    analyzer.cluster_spectrum(N_CLUSTERS_FIXED)
    labels = np.array(analyzer.labels)
    uniq, counts = np.unique(labels, return_counts=True)

    ignored_cluster = int(uniq[np.argmin(counts)])
    loop_labels = [int(l) for l in uniq if l != ignored_cluster]

    for lbl in loop_labels:
        L0 = max(1, int(analyzer._default_cycle_length(lbl)))
        for L in range(L0, MAX_CYCLE_LENGTH + 1):
            # This fills analyzer._mi_cache[(lbl, L)] if not present
            analyzer._get_cached_mi_results(lbl, L)

    payload = {
        "ham_digest": ham_digest,
        "n_clusters": N_CLUSTERS_FIXED,
        "labels": analyzer.labels,
        "mi_cache": analyzer._mi_cache,
        "ignored_cluster": ignored_cluster,
        "max_cycle_length": MAX_CYCLE_LENGTH,
    }

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(payload, f)

    analyzer._ignored_cluster_label = ignored_cluster
    analyzer._max_cycle_length = MAX_CYCLE_LENGTH


@st.cache_resource
def get_analyzer() -> DualCycleAnalyzer:
    """Streamlit-cached DualCycleAnalyzer with NMI cache preloaded."""
    if not HAM_PATH.exists():
        raise FileNotFoundError(
            f"Hamiltonian CSV not found at {HAM_PATH}. "
            "Place your Ham_96.csv there or update HAM_PATH."
        )

    df = pd.read_csv(HAM_PATH, header=None)
    H = df.values
    analyzer = DualCycleAnalyzer(H)

    _load_or_build_nmi_cache(analyzer)
    return analyzer


# ---------------------------------------------------------------------
# Figure builder helpers
# ---------------------------------------------------------------------
def _add_spectral_panel(
    fig,
    analyzer: DualCycleAnalyzer,
    target_cluster: int,
    row: int,
    col: int,
) -> None:
    """Custom spectral panel that visually separates isolated modes."""
    labels = np.array(analyzer.labels)
    uniq = np.unique(labels)
    ignored = getattr(analyzer, "_ignored_cluster_label", None)
    evals = analyzer.evals

    for l in uniq:
        mask = labels == l
        if not np.any(mask):
            continue

        if ignored is not None and int(l) == int(ignored):
            name = "Isolated modes"
            opacity = 0.4
            marker = dict(size=6, color="#bbbbbb")
            legendgroup = "isolated"
        else:
            name = f"Loop {int(l)}"
            opacity = 1.0 if int(l) == int(target_cluster) else 0.25
            marker = dict(size=7)
            legendgroup = f"loop_{int(l)}"

        fig.add_trace(
            {
                "type": "scatter",
                "x": np.real(evals[mask]),
                "y": np.imag(evals[mask]),
                "mode": "markers",
                "marker": marker,
                "opacity": float(opacity),
                "name": name,
                "legendgroup": legendgroup,
                "hovertemplate": (
                    "Re(E)=%{x:.4f}<br>Im(E)=%{y:.4f}"
                    f"<extra>{name}</extra>"
                ),
            },
            row=row,
            col=col,
        )


def build_dashboard_figure(
    analyzer: DualCycleAnalyzer,
    target_cluster: int,
    stat_type: str,
    layout_algo: str,
    mi_rank: int,
    cycle_length_input: int,
) -> Tuple[Any, Dict[str, Any]]:
    """2×2 layout:
        Row 1: [ spectral loops ] (colspan=2)
        Row 2: [ lattice stats | top NMI cycle ]
    """
    # Ensure clustering fixed at k=3 (but do NOT recluster if already done)
    if analyzer.labels is None or getattr(analyzer, "n_clusters", None) != N_CLUSTERS_FIXED:
        analyzer.cluster_spectrum(N_CLUSTERS_FIXED)

    labels_full = np.array(analyzer.labels)
    uniq = np.unique(labels_full)
    ignored = getattr(analyzer, "_ignored_cluster_label", None)

    loop_labels = [int(l) for l in uniq if ignored is None or int(l) != int(ignored)]
    if not loop_labels:
        loop_labels = [int(uniq[0])]
    if target_cluster not in loop_labels:
        target_cluster = loop_labels[0]

    # Real-space lattice stats for chosen loop
    node_w, _, all_edges, edge_w_und = analyzer._prepare_real_space_stats(
        target_cluster, layout_algo
    )

    # Cycle length: 0 => default, otherwise clamp
    if cycle_length_input <= 0:
        cycle_length = None
    else:
        cycle_length = int(max(1, min(cycle_length_input, MAX_CYCLE_LENGTH)))

    mi_results = analyzer._get_cached_mi_results(target_cluster, cycle_length)
    top_k = len(mi_results)
    if top_k > 0:
        rank_idx = max(0, min(mi_rank - 1, top_k - 1))
        sel = mi_results[rank_idx]
        cyc = sel["cycle"]
        nmi_score = float(sel["nmi_score"])
        total_weight = float(sel.get("total_weight", 0.0))
        cycle_length_effective = len(cyc)
    else:
        rank_idx = 0
        cyc = []
        nmi_score = 0.0
        total_weight = 0.0
        cycle_length_effective = 0

    # 2×2 grid
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Complex Eigenvalues (Spectral Loops)",
            # "",
            "Real-Space Lattice (Biorthogonal Weights)",
            "Top NMI cycle",
        ),
        specs=[
            [{"type": "scatter", "colspan": 2}, None],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
    )

    # Row 1: spectral loops (square aspect)
    _add_spectral_panel(fig, analyzer, target_cluster, row=1, col=1)
    fig.update_xaxes(title_text="Re(E)", row=1, col=1)
    fig.update_yaxes(
        title_text="Im(E)",
        scaleanchor="x", scaleratio=1,
        row=1, col=1,
    )

    # Row 2, col 1: lattice stats, compact colourbar on the left of this row only
    analyzer._add_lattice_stat_traces(
        fig,
        stat_type=stat_type,
        node_w=node_w,
        edge_w_und=edge_w_und,
        all_edges=all_edges,
        colorscale="dense",
        row=2,
        col=1,
        show_colorbar=True,
        colorbar_x=0.42,   # left of subplot
        colorbar_y=0.22,   # roughly centred on 2nd row
        colorbar_len=0.30, # shorter colourbar
    )
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)

    # Row 2, col 2: top NMI cycle
    analyzer._add_cycle_panel_background(fig, row=2, col=2)
    if cyc:
        analyzer._add_cycle_highlight(
            fig,
            cyc,
            row=2,
            col=2,
            name=f"NMI cycle (rank {rank_idx + 1})",
        )
        if len(fig.layout.annotations) >= 4:
            fig.layout.annotations[3].text = (
                f"Top NMI cycle (L={cycle_length_effective}, "
                f"rank {rank_idx + 1}/{top_k}, NMI={nmi_score:.3g})"
            )
    elif len(fig.layout.annotations) >= 4:
        fig.layout.annotations[3].text = "Top NMI cycle (none)"

    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)

    fig.update_layout(
        height=850,
        width=800,
        autosize=True,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            title_text="Spectral loops",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#eaeaea",
            borderwidth=2,
        ),
    )

    cluster_size = int((labels_full == target_cluster).sum())
    meta = {
        "target_cluster": int(target_cluster),
        "cluster_size": cluster_size,
        "nmi_score": float(nmi_score),
        "cycle_length": int(cycle_length_effective),
        "rank": int(rank_idx + 1) if top_k > 0 else 0,
        "top_k": int(top_k),
        "total_weight": float(total_weight),
    }
    return fig, meta


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Dual Cycle Analyzer – Spectral Loops vs Real-Space Cycles",
        layout="wide",
    )

    st.title("Reveal Spectral Loops — Real-Space Cycles Duality with Biorthogonal Weights & Normalized Mutual Information")

    render_theory_intro()

    # Load analyzer (with cache) once per session
    analyzer = get_analyzer()
    labels_full = np.array(analyzer.labels)
    uniq = np.unique(labels_full)
    ignored = getattr(analyzer, "_ignored_cluster_label", None)
    loop_labels = [int(l) for l in uniq if ignored is None or int(l) != int(ignored)]

    # Sidebar controls
    st.sidebar.header("Controls")

    if not loop_labels:
        loop_labels = [int(uniq[0])]

    # ----- Session state bookkeeping -----
    if "target_cluster" not in st.session_state:
        st.session_state["target_cluster"] = loop_labels[0]
    if "prev_target_cluster" not in st.session_state:
        st.session_state["prev_target_cluster"] = st.session_state["target_cluster"]

    # Initialize cycle length and NMI rank defaults for the initial target loop
    if "cycle_length" not in st.session_state or "mi_rank" not in st.session_state:
        initial_cluster = st.session_state["target_cluster"]
        default_L, default_rank = _get_ui_defaults_for_loop(initial_cluster, labels_full)
        st.session_state["cycle_length"] = default_L
        st.session_state["mi_rank"] = default_rank

    current_target = st.session_state.get("target_cluster", loop_labels[0])
    if current_target not in loop_labels:
        current_target = loop_labels[0]
        st.session_state["target_cluster"] = current_target

    # Target loop dropdown (backed by session_state["target_cluster"])
    target_cluster = st.sidebar.selectbox(
        "Target spectral loop",
        options=loop_labels,
        index=loop_labels.index(current_target),
        key="target_cluster",
        help=(
            "Choose which spectral loop (cluster) to analyze. "
            "The smallest-count cluster is treated as isolated modes and excluded."
        ),
    )

    # If the target loop changed, reset L and NMI rank to loop-dependent defaults
    if st.session_state["target_cluster"] != st.session_state["prev_target_cluster"]:
        new_cluster = st.session_state["target_cluster"]
        default_L, default_rank = _get_ui_defaults_for_loop(new_cluster, labels_full)
        st.session_state["cycle_length"] = default_L
        st.session_state["mi_rank"] = default_rank
        st.session_state["prev_target_cluster"] = new_cluster

    stat_type = st.sidebar.radio(
        "Lattice metric",
        options=["edge_centric", "node_centric"],
        index=0,
        help=(
            "Edge-centric: colour and thicken edges by spectral weight.\n"
            "Node-centric: scale node size / colour by spectral weight."
        ),
    )

    layout_algo = st.sidebar.radio(
        "Layout algorithm",
        options=["kamada_kawai", "planar"],
        index=0,
        help="Graph layout for the real-space lattice.",
    )

    # Cycle length controlled by session_state["cycle_length"]
    cycle_length_input = st.sidebar.number_input(
        "Cycle length L (0 = default, max 88)",
        min_value=0,
        max_value=MAX_CYCLE_LENGTH,
        value=int(st.session_state["cycle_length"]),
        step=1,
        key="cycle_length",
        help=(
            "Specify real-space cycle length for the MI/NMI analysis. "
            "L = 0 uses the analyzer's internal default (typically the size of the spectral loop). "
            f"NMI cycles are pre-computed for L in [L_default, {MAX_CYCLE_LENGTH}] "
            "for each spectral loop (excluding the smallest cluster)."
        ),
    )

    # Pre-fetch NMI results to size the rank control (no recompute; uses cache)
    cycle_length_for_slider = None if cycle_length_input <= 0 else int(cycle_length_input)
    mi_results = analyzer._get_cached_mi_results(target_cluster, cycle_length_for_slider)
    num_results = len(mi_results)

    if num_results <= 1:
        # No slider if there are 0 or 1 cycles: fix rank = 1
        mi_rank = 1
        st.session_state["mi_rank"] = mi_rank
        if num_results == 0:
            st.sidebar.info(
                "No simple cycles found for this spectral loop and cycle length L. "
                "NMI rank is fixed to 1."
            )
        else:
            st.sidebar.info(
                "Only one simple cycle found for this spectral loop and L. "
                "NMI rank is fixed to 1."
            )
    else:
        # Clamp stored mi_rank to [1, num_results]
        stored_rank = int(st.session_state.get("mi_rank", 1))
        if stored_rank < 1 or stored_rank > num_results:
            stored_rank = max(1, min(stored_rank, num_results))
            st.session_state["mi_rank"] = stored_rank

        mi_rank = st.sidebar.slider(
            "NMI cycle rank",
            min_value=1,
            max_value=num_results,
            value=stored_rank,
            key="mi_rank",
            help="Rank cycles by normalized mutual information (1 = highest NMI).",
        )

    # Build figure and metadata
    fig, meta = build_dashboard_figure(
        analyzer=analyzer,
        target_cluster=target_cluster,
        stat_type=stat_type,
        layout_algo=layout_algo,
        mi_rank=mi_rank,
        cycle_length_input=int(cycle_length_input),
    )

    # Main layout: embed Plotly HTML in a scrollable container so the
    # dashboard keeps its native size and aspect ratio. If the dashboard is
    # wider than the viewport, the browser shows a horizontal scroll bar.
    layout_width = fig.layout.width or 800
    layout_height = fig.layout.height or 850

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        default_width=f"{layout_width}px",
        default_height=f"{layout_height}px",
    )

    components.html(
        f"""
        <div style="width: 100%; overflow-x: auto;">
            {plot_html}
        </div>
        """,
        height=layout_height,
        scrolling=True,
    )

    # Info panel
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Target loop label", f"{meta['target_cluster']}")
        st.metric("Eigenvalues in loop", f"{meta['cluster_size']}")

    with c2:
        st.metric("NMI rank", f"{meta['rank']} / {meta['top_k']}")
        st.metric("Cycle length L", f"{meta['cycle_length']}")

    with c3:
        st.metric("NMI score", f"{meta['nmi_score']:.3g}")
        st.metric("Total edge weight", f"{meta['total_weight']:.3g}")

    # Optional: table of top NMI cycles (current loop / L)
    with st.expander("Show top NMI cycles table (current loop / L)"):
        table_rows = []
        for rank_idx, item in enumerate(mi_results, start=1):
            table_rows.append(
                {
                    "rank": rank_idx,
                    "L": len(item["cycle"]),
                    "NMI": float(item["nmi_score"]),
                    "total_weight": float(item.get("total_weight", 0.0)),
                    "cycle": item["cycle"],
                }
            )
        if table_rows:
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table)
        else:
            st.info("No cycles found for this loop / cycle length.")

    section_conceptual_diagrams()


if __name__ == "__main__":
    main()