import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
from scipy.linalg import eig
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import ipywidgets as widgets
from IPython.display import display
from typing import Tuple, List, Dict, Optional


class DualCycleAnalyzer:
    """Relate spectral loops of a non-Hermitian Hamiltonian to real-space cycles."""

    def __init__(self, hamiltonian: np.ndarray):
        """hamiltonian: (N, N) non-Hermitian Hamiltonian / adjacency matrix."""
        self.H = np.array(hamiltonian)
        self.N = self.H.shape[0]

        self.evals, self.VL, self.VR = eig(self.H, left=True, right=True)
        vl_norms = np.linalg.norm(self.VL, axis=0)
        vr_norms = np.linalg.norm(self.VR, axis=0)
        vl_norms[vl_norms == 0] = 1.0
        vr_norms[vr_norms == 0] = 1.0
        self.VL /= vl_norms
        self.VR /= vr_norms

        order = np.lexsort((np.abs(self.evals), np.angle(self.evals)))
        self.evals = self.evals[order]
        self.VL = self.VL[:, order]
        self.VR = self.VR[:, order]

        self.DiG = nx.from_numpy_array(self.H, create_using=nx.DiGraph)
        self.G = self.DiG.to_undirected()

        self.pos: Dict[int, np.ndarray] = {}
        self._current_layout_algo: Optional[str] = None

        self.labels: Optional[np.ndarray] = None
        self.n_clusters: int = 0

        self.last_mi_analysis: Optional[Dict] = None
        self._mi_cache: Dict[Tuple[int, int], List[Dict]] = {}

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def compute_layout(self, algorithm: str = "kamada_kawai") -> Dict[int, np.ndarray]:
        if self._current_layout_algo == algorithm and self.pos:
            return self.pos
        if algorithm == "planar":
            try:
                self.pos = nx.planar_layout(self.G)
            except nx.NetworkXException:
                # Fall back gracefully if the graph is not planar
                self.pos = nx.kamada_kawai_layout(self.G, weight=None)
        else:
            self.pos = nx.kamada_kawai_layout(self.G, weight=None)
        self._current_layout_algo = algorithm
        return self.pos

    def cluster_spectrum(self, n_clusters: int = 3) -> np.ndarray:
        self.n_clusters = int(n_clusters)
        self._mi_cache.clear()

        # r = np.abs(self.evals).reshape(-1, 1)
        re = self.evals.real
        im = self.evals.imag
        re /= np.max(np.abs(re)) if np.max(np.abs(re)) > 0 else 1.0
        im /= np.max(np.abs(im)) if np.max(np.abs(im)) > 0 else 1.0
        r = np.abs(re + 1j * im).reshape(-1, 1)
        
        self.labels = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42).fit_predict(r)
        return self.labels

    def _default_cycle_length(self, target_loop_label: int) -> int:
        if self.labels is None:
            self.cluster_spectrum()
        return int(np.sum(self.labels == target_loop_label))

    def _ensure_clustering(self, n_clusters: int, target_cluster: int) -> Tuple[int, np.ndarray]:
        if self.labels is None or self.n_clusters != n_clusters:
            self.cluster_spectrum(n_clusters)
        labels = np.unique(self.labels)
        if target_cluster not in labels:
            target_cluster = int(labels[0])
        return int(target_cluster), labels

    def _get_projector_weights(self, label: int) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        if self.labels is None:
            raise RuntimeError("cluster_spectrum(...) must be called before projector weights.")
        idx = np.where(self.labels == label)[0]
        if idx.size == 0:
            raise ValueError(f"Spectral cluster {label} has no members.")
        node_w = (np.abs(self.VL[:, idx]) * np.abs(self.VR[:, idx])).sum(axis=1).astype(float)

        rows, cols = np.where(self.H != 0)
        absH = np.abs(self.H)
        edge_w: Dict[Tuple[int, int], float] = {}
        for i, j in zip(rows, cols):
            h = float(absH[i, j])
            if h == 0.0:
                continue
            edge_w[(i, j)] = h * float((np.abs(self.VL[i, idx]) * np.abs(self.VR[j, idx])).sum())
        return node_w, edge_w

    def real_space_stats(self, target_loop_label: int) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        if self.labels is None:
            self.cluster_spectrum(3)
        return self._get_projector_weights(int(target_loop_label))

    # ------------------------------------------------------------------
    # Cycle enumeration (undirected view)
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_cycle(cycle: List[int]) -> Tuple[int, ...]:
        if not cycle:
            return tuple()
        c = list(cycle)
        n = len(c)
        rots = [tuple(c[i:] + c[:i]) for i in range(n)]
        rc = list(reversed(c))
        rots_rev = [tuple(rc[i:] + rc[:i]) for i in range(n)]
        return min(rots + rots_rev)

    def enumerate_real_cycles(self, length: int) -> List[List[int]]:
        try:
            it = nx.simple_cycles(self.G)  # type: ignore[arg-type]
        except Exception:
            dig = nx.DiGraph()
            dig.add_edges_from(self.G.edges())
            it = nx.simple_cycles(dig)
        seen, out = set(), []
        for c in it:
            if len(c) != length:
                continue
            key = self._canonical_cycle(c)
            if key not in seen:
                seen.add(key)
                out.append(list(key))
        return out

    # ------------------------------------------------------------------
    # Mutual information
    # ------------------------------------------------------------------
    def calculate_mutual_information(self, target_loop_label: int, cycle_length: Optional[int] = None) -> List[Dict]:
        """Mutual information between one spectral loop and all real-space cycles of length L.

        The returned scores are mapped to a correlation-like NMI in [0, 1) via
            NMI = sqrt(1 - exp(-2 * MI))
        applied per cycle.
        """
        if self.labels is None:
            self.cluster_spectrum()
        if cycle_length is None:
            cycle_length = self._default_cycle_length(target_loop_label)
        node_w, edge_w_dir = self._get_projector_weights(int(target_loop_label))

        all_edges = list(self.G.edges())
        edge_to_idx = {e: i for i, e in enumerate(all_edges)}
        Y = np.array(
            [edge_w_dir.get((u, v), 0.0) + edge_w_dir.get((v, u), 0.0) for u, v in all_edges],
            dtype=float,
        )

        cycles = self.enumerate_real_cycles(int(cycle_length))
        results: List[Dict] = []
        for cyc in cycles:
            mask = np.zeros(len(all_edges), dtype=int)
            for u, v in zip(cyc, cyc[1:] + cyc[:1]):
                idx = edge_to_idx.get((u, v))
                if idx is None:
                    idx = edge_to_idx.get((v, u))
                if idx is not None:
                    mask[idx] = 1
            if mask.sum() == 0:
                mi = 0.0
            else:
                mi = float(
                    mutual_info_regression(
                        mask.reshape(-1, 1),
                        Y,
                        discrete_features=[0],
                        random_state=42,
                    )[0]
                )
            total_w = float(Y[mask == 1].sum())
            results.append({"cycle": cyc, "mi_raw": mi, "total_weight": total_w})

        # Map raw MI to correlation-like NMI in [0, 1)
        for r in results:
            mi = max(r["mi_raw"], 0.0)
            val = 1.0 - np.exp(-2.0 * mi)
            val = min(max(val, 0.0), 1.0)  # numerical safety
            r["nmi_score"] = float(np.sqrt(val))


        results.sort(key=lambda x: x["nmi_score"], reverse=True)
        self.last_mi_analysis = {
            "target_loop_label": int(target_loop_label),
            "cycle_length": int(cycle_length),
            "node_weights": node_w,
            "directed_edge_weights": edge_w_dir,
            "undirected_edges": all_edges,
            "edge_to_index": edge_to_idx,
            "edge_spectral_weights": Y,
            "cycles": cycles,
            "results": results,
            "normalized": True,
        }
        return results

    def _get_cached_mi_results(self, target_loop_label: int, cycle_length: Optional[int] = None) -> List[Dict]:
        if self.labels is None:
            self.cluster_spectrum()
        if cycle_length is None:
            cycle_length = self._default_cycle_length(target_loop_label)
        key = (int(target_loop_label), int(cycle_length))
        if key not in self._mi_cache:
            self._mi_cache[key] = self.calculate_mutual_information(target_loop_label, cycle_length)
        return self._mi_cache[key]

    def _prime_mi_cache(self, labels: np.ndarray, show_progress: bool = False) -> None:
        uniq = list(dict.fromkeys(int(l) for l in labels))
        progress = None
        if show_progress and len(uniq) > 1:
            try:
                progress = widgets.IntProgress(value=0, min=0, max=len(uniq), description="MI cache", bar_style="info")
                display(progress)
            except Exception:
                progress = None
        for i, lbl in enumerate(uniq, 1):
            self._get_cached_mi_results(lbl, None)
            if progress:
                progress.value = i
        if progress:
            progress.bar_style = "success"
            progress.description = "MI cycle cached"
        elif show_progress:
            print(f"Cached MI for {len(uniq)} loop(s).")

    # ------------------------------------------------------------------
    # Plotly helpers
    # ------------------------------------------------------------------
    def _prepare_real_space_stats(
        self, target_cluster: int, layout_algo: str
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], List[Tuple[int, int]], np.ndarray]:
        self.compute_layout(layout_algo)
        node_w, edge_w_dir = self._get_projector_weights(target_cluster)
        all_edges = list(self.G.edges())
        edge_w_und = np.array(
            [edge_w_dir.get((u, v), 0.0) + edge_w_dir.get((v, u), 0.0) for u, v in all_edges],
            dtype=float,
        )
        return node_w, edge_w_dir, all_edges, edge_w_und

    def _add_spectral_traces(self, fig: go.Figure, labels: np.ndarray, target_cluster: int, row: int, col: int) -> None:
        for l in labels:
            mask = self.labels == l
            fig.add_trace(
                go.Scatter(
                    x=np.real(self.evals[mask]),
                    y=np.imag(self.evals[mask]),
                    mode="markers",
                    marker=dict(size=7),
                    opacity=1.0 if l == target_cluster else 0.15,
                    name=f"Loop {l}",
                    hovertemplate="Re(E)=%{x:.4f}<br>Im(E)=%{y:.4f}<extra>Cluster {l}</extra>",
                ),
                row=row,
                col=col,
            )

    def _add_lattice_stat_traces(
        self,
        fig: go.Figure,
        *,
        stat_type: str,
        node_w: np.ndarray,
        edge_w_und: np.ndarray,
        all_edges: List[Tuple[int, int]],
        colorscale: str,
        row: int,
        col: int,
        show_colorbar: bool = True,
        colorbar_x: Optional[float] = None,
        colorbar_y: Optional[float] = None,
        colorbar_len: Optional[float] = None,
    ) -> None:
        """Add either edge-centric or node-centric lattice statistics to a subplot.

        The colorbar can be repositioned using colorbar_x / colorbar_y / colorbar_len
        in figure-fraction coordinates.
        """
        bg_x, bg_y = [], []
        for u, v in all_edges:
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            bg_x.extend([x0, x1, None])
            bg_y.extend([y0, y1, None])
        fig.add_trace(
            go.Scatter(
                x=bg_x,
                y=bg_y,
                mode="lines",
                line=dict(color="#dddddd", width=1),
                hoverinfo="none",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Compact colourbar settings
        cb_x = float(colorbar_x) if colorbar_x is not None else 1.02
        cb_y = float(colorbar_y) if colorbar_y is not None else 0.5
        cb_len = float(colorbar_len) if colorbar_len is not None else 0.4

        if stat_type == "edge_centric" and len(all_edges) > 0:
            e_min, e_max = float(edge_w_und.min()), float(edge_w_und.max())
            if e_max <= e_min:
                e_max = e_min + 1.0
            norm_vals = (edge_w_und - e_min) / (e_max - e_min)
            widths = 1.5 + 4.0 * norm_vals
            colors = sample_colorscale(colorscale, norm_vals.tolist())
            for (u, v), w, color, width in zip(all_edges, edge_w_und, colors, widths):
                if w <= 0.0:
                    continue
                x0, y0 = self.pos[u]
                x1, y1 = self.pos[v]
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(color=color, width=float(width)),
                        hoverinfo="text",
                        text=f"Edge ({u}, {v})<br>W = {w:.4e}",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Invisible marker that only hosts the colourbar
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=0,
                        color=edge_w_und,
                        colorscale=colorscale,
                        showscale=show_colorbar,
                        colorbar=dict(
                            title="Edge weight",
                            x=cb_x,
                            y=cb_y,
                            len=cb_len,
                        ),
                    ),
                    hoverinfo="none",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        node_x = [self.pos[i][0] for i in range(self.N)]
        node_y = [self.pos[i][1] for i in range(self.N)]
        if stat_type == "node_centric":
            max_w = float(np.max(node_w)) if np.max(node_w) > 0 else 1.0
            sizes = 2.5 + 4 * (node_w / max_w)
            marker = dict(
                size=sizes,
                color=node_w,
                colorscale=colorscale,
                showscale=show_colorbar,
                colorbar=dict(
                    title="Node weight",
                    x=cb_x,
                    y=cb_y,
                    len=cb_len,
                ),
                line=dict(color="#333333", width=0.25),
            )
        else:
            marker = dict(size=6, color="#989898")

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=marker,
                text=[f"Node {i}<br>w_i = {node_w[i]:.4e}" for i in range(self.N)],
                hoverinfo="text",
                name="Lattice nodes",
                showlegend=stat_type == "node_centric",
            ),
            row=row,
            col=col,
        )

    def _add_cycle_highlight(self, fig: go.Figure, cycle_nodes: List[int], row: int, col: int, name: str = "Cycle") -> None:
        edges = list(zip(cycle_nodes, cycle_nodes[1:] + cycle_nodes[:1]))
        hx, hy, labels = [], [], []
        for u, v in edges:
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            hx.extend([x0, x1, None])
            hy.extend([y0, y1, None])
            labels.extend([f"{name} edge ({u}, {v})", f"{name} edge ({u}, {v})", None])
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="lines",
                line=dict(color="#111111", width=5),
                opacity=0.85,
                hoverinfo="text",
                text=labels,
                name=name,
                legendgroup=name,
                showlegend=True,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[self.pos[i][0] for i in cycle_nodes],
                y=[self.pos[i][1] for i in cycle_nodes],
                mode="markers",
                marker=dict(size=6, color="#ffb703", line=dict(color="#111111", width=2)),
                hoverinfo="text",
                text=[f"{name} node {i}" for i in cycle_nodes],
                name=name,
                legendgroup=name,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _add_cycle_panel_background(self, fig: go.Figure, row: int, col: int) -> None:
        bg_x, bg_y = [], []
        for u, v in self.G.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            bg_x.extend([x0, x1, None])
            bg_y.extend([y0, y1, None])
        fig.add_trace(
            go.Scatter(
                x=bg_x,
                y=bg_y,
                mode="lines",
                line=dict(color="#e0e0e0", width=1),
                hoverinfo="none",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[self.pos[i][0] for i in range(self.N)],
                y=[self.pos[i][1] for i in range(self.N)],
                mode="markers",
                marker=dict(size=6, color="#b0b0b0"),
                hoverinfo="text",
                text=[f"Node {i}" for i in range(self.N)],
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def plot_lattice_weights(
        self,
        n_clusters: int = 3,
        target_cluster: int = 0,
        stat_type: str = "node_centric",
        layout_algo: str = "kamada_kawai",
    ) -> go.Figure:
        """Static 2-panel view: spectrum + real-space weights for one loop."""
        target_cluster, labels = self._ensure_clustering(n_clusters, target_cluster)
        node_w, _, all_edges, edge_w_und = self._prepare_real_space_stats(target_cluster, layout_algo)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Complex Eigenvalues (Spectral Loops)", "Real‑Space Lattice"),
            horizontal_spacing=0.05,
            specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        )
        self._add_spectral_traces(fig, labels, target_cluster, row=1, col=1)
        fig.update_xaxes(title_text="Re(E)", row=1, col=1)
        fig.update_yaxes(title_text="Im(E)", row=1, col=1)
        self._add_lattice_stat_traces(
            fig,
            stat_type=stat_type,
            node_w=node_w,
            edge_w_und=edge_w_und,
            all_edges=all_edges,
            colorscale="dense",
            row=1,
            col=2,
        )
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        fig.update_layout(
            height=600,
            width=1300,
            template="plotly_white",
            showlegend=True,
            title_text=f"Spectral analysis – cluster {target_cluster}",
            legend=dict(
                x=-0.05,
                y=1.0,
                xanchor="right",
                yanchor="top",
                title_text="Spectral loops",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#eaeaea",
                borderwidth=1,
            ),
        )
        return fig

    def plot_cycle_by_mutual_information(
        self,
        target_loop_label: int,
        rank: int = 0,
        top_k: Optional[int] = None,
        cycle_length: Optional[int] = None,
        stat_type: str = "edge_centric",
        layout_algo: str = "kamada_kawai",
        n_clusters: Optional[int] = None,
    ) -> go.Figure:
        n_clusters = n_clusters if n_clusters is not None else (self.n_clusters or 3)
        target_loop_label, labels = self._ensure_clustering(n_clusters, target_loop_label)
        fig = self.plot_lattice_weights(
            n_clusters=n_clusters,
            target_cluster=target_loop_label,
            stat_type=stat_type,
            layout_algo=layout_algo,
        )
        cycle_length_val = None if (cycle_length is None or int(cycle_length) <= 0) else int(cycle_length)
        mi_results = self._get_cached_mi_results(target_loop_label, cycle_length_val)
        if top_k is not None:
            mi_results = mi_results[: max(1, int(top_k))]
        if not mi_results:
            return fig
        rank = max(0, min(int(rank), len(mi_results) - 1))
        sel = mi_results[rank]
        cyc = sel["cycle"]
        if cyc:
            self._add_cycle_highlight(fig, cyc, row=1, col=2, name=f"NMI cycle (rank {rank + 1})")
        nmi_score = sel["nmi_score"]
        total_w = sel.get("total_weight", 0.0)
        base_title = fig.layout.title.text or "Spectral analysis"
        fig.update_layout(
            title_text=f"{base_title} | NMI rank {rank + 1}/{len(mi_results)} (NMI={nmi_score:.3g}, weight={total_w:.3g})"
        )
        return fig

    def interactive_view(self, default_cluster: int) -> widgets.interactive:
        """Interactive spectrum / real-space / top-NMI-cycle dashboard in Jupyter."""
        if self.labels is None:
            self.cluster_spectrum(3)
        self._prime_mi_cache(np.unique(self.labels), show_progress=True)

        w_clusters = widgets.IntSlider(value=self.n_clusters or 3, min=2, max=7, description="KMeans k")
        w_cycle_len = widgets.IntText(value=0, description="Cycle L (0=auto)")

        def _initial_top_k(target_cluster: int, n_clusters: Optional[int] = None, cycle_length: Optional[int] = None) -> int:
            n_clusters = n_clusters if n_clusters is not None else w_clusters.value
            tgt, labels = self._ensure_clustering(n_clusters, target_cluster)
            self._prime_mi_cache(labels, show_progress=False)
            cl = None if (cycle_length is None or int(cycle_length) <= 0) else int(cycle_length)
            mi_res = self._get_cached_mi_results(tgt, cl)
            return max(1, min(len(mi_res), 15)) if mi_res else 1

        w_target = widgets.Dropdown(
            options=list(range(w_clusters.value)),
            value=min(default_cluster, w_clusters.value - 1),
            description="Target loop",
        )
        w_stat = widgets.Select(
            options=["edge_centric", "node_centric"],
            value="edge_centric",
            description="Metric",
            rows=2,
        )
        w_layout = widgets.Select(
            options=["kamada_kawai", "planar"],
            value="kamada_kawai",
            description="Layout",
            rows=2,
        )
        w_mi_rank = widgets.IntSlider(
            value=1,
            min=1,
            max=_initial_top_k(w_target.value, None, w_cycle_len.value),
            step=1,
            description="Cycle rank",
        )

        def on_cluster_change(change):
            k = change["new"]
            w_target.options = list(range(k))
            if w_target.value >= k:
                w_target.value = 0
            w_mi_rank.max = _initial_top_k(w_target.value, k, w_cycle_len.value)

        def on_target_change(change):
            w_mi_rank.max = _initial_top_k(change["new"], None, w_cycle_len.value)
            if w_mi_rank.value > w_mi_rank.max:
                w_mi_rank.value = w_mi_rank.max

        def on_cycle_len_change(change):
            w_mi_rank.max = _initial_top_k(w_target.value, None, change["new"])
            if w_mi_rank.value > w_mi_rank.max:
                w_mi_rank.value = w_mi_rank.max

        w_clusters.observe(on_cluster_change, names="value")
        w_target.observe(on_target_change, names="value")
        w_cycle_len.observe(on_cycle_len_change, names="value")

        def _update(n_clusters, target_cluster, stat_type, layout_algo, mi_rank, cycle_length):
            target_cluster, labels = self._ensure_clustering(n_clusters, target_cluster)
            self._prime_mi_cache(labels, show_progress=False)
            node_w, _, all_edges, edge_w_und = self._prepare_real_space_stats(target_cluster, layout_algo)

            cl = None if (cycle_length is None or int(cycle_length) <= 0) else int(cycle_length)
            mi_results = self._get_cached_mi_results(target_cluster, cl)
            if mi_results:
                top_k = min(len(mi_results), 15)
                w_mi_rank.max = max(top_k, 1)
                rank_idx = min(max(mi_rank, 1), w_mi_rank.max) - 1
                sel = mi_results[rank_idx]
                cyc = sel["cycle"]
                nmi_score = sel["nmi_score"]
            else:
                w_mi_rank.max = 1
                rank_idx, cyc, nmi_score = 0, [], 0.0

            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    "Complex Eigenvalues (Spectral Loops)",
                    "Real‑Space Lattice",
                    "Top NMI cycle",
                ),
                horizontal_spacing=0.04,
                specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
            )

            self._add_spectral_traces(fig, labels, target_cluster, row=1, col=1)
            fig.update_xaxes(title_text="Re(E)", row=1, col=1)
            fig.update_yaxes(title_text="Im(E)", row=1, col=1)

            self._add_lattice_stat_traces(
                fig,
                stat_type=stat_type,
                node_w=node_w,
                edge_w_und=edge_w_und,
                all_edges=all_edges,
                colorscale="dense",
                row=1,
                col=2,
                show_colorbar=True,
            )
            fig.update_xaxes(visible=False, row=1, col=2)
            fig.update_yaxes(visible=False, row=1, col=2)

            self._add_cycle_panel_background(fig, row=1, col=3)
            if cyc:
                self._add_cycle_highlight(fig, cyc, row=1, col=3, name=f"NMI cycle (rank {rank_idx + 1})")
                if len(fig.layout.annotations) >= 3:
                    fig.layout.annotations[2].text = (
                        f"Top NMI cycle (L={cycle_length if int(cycle_length) > 0 else 'auto'}, "
                        f"rank {rank_idx + 1}/{w_mi_rank.max}, NMI={nmi_score:.3g})"
                    )
            elif len(fig.layout.annotations) >= 3:
                fig.layout.annotations[2].text = "Top NMI cycle (none)"

            fig.update_xaxes(visible=False, row=1, col=3)
            fig.update_yaxes(visible=False, row=1, col=3)

            fig.update_layout(
                height=600,
                width=1600,
                template="plotly_white",
                showlegend=True,
                title_text=f"Spectral analysis – cluster {target_cluster}",
                legend=dict(
                    x=-0.05,
                    y=1.0,
                    xanchor="right",
                    yanchor="top",
                    title_text="Spectral loops",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#eaeaea",
                    borderwidth=1,
                ),
            )
            return fig

        return widgets.interact(
            _update,
            n_clusters=w_clusters,
            target_cluster=w_target,
            stat_type=w_stat,
            layout_algo=w_layout,
            mi_rank=w_mi_rank,
            cycle_length=w_cycle_len,
        )


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/Ham_96.csv", header=None)
    H = df.values
    analyzer = DualCycleAnalyzer(H)
    analyzer.interactive_view(default_cluster=2)

    mi_results = analyzer.calculate_mutual_information(target_loop_label=2, cycle_length=None)
    print("Top 5 cycles by NMI score:")
    for res in mi_results[:5]:
        print(res)