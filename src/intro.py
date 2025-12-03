import streamlit as st
from streamlit_mermaid import st_mermaid


def _section_physical_context_and_method() -> None:
    st.markdown("### Ⅰ. Physical Context and Method")
    with st.expander("Click here to expand", expanded=False):
        st.markdown(
            r"""
### Overview

This app explores a **spectral-loop ↔ lattice-cycle correspondence** in a non-Hermitian hyperbolic lattice.

The non-Hermitian Hamiltonian $H$ has complex eigenvalues that form **loops** in the complex-energy plane (“spectral loops”). For each such loop we:

1. Build a **biorthogonal projector** from left/right eigenvectors.
2. Extract **node** and **edge** “biorthogonal weights” in real space and sum over all modes in a chosen spectral loop, which can be seen as a *spatial profile* of that spectral loop.
3. Enumerate simple cycles of the real-space lattice.
4. Rank lattice cycles by how well their edge set “aligns” with the spectral weights, using a **normalized mutual information (NMI)** metric.

The goal is to make precise, in a lattice setting more complicated than Hatano–Nelson, how a **spectral loop** $\Gamma_\ell$ can correspond to a **real-space cycle** $C_\ell$.

---

### Hamiltonian and spectral loops

We start from a finite non-Hermitian hyperbolic Hamiltonian
$$
H \in \mathbb{C}^{N\times N},
$$
which one can think of as the adjacency matrix of a directed graph on $N$ sites.

We diagonalize $H$ biorthogonally:
$$
H \ket{\psi_R^{(n)}} = E_n \ket{\psi_R^{(n)}}, \qquad
\bra{\psi_L^{(n)}} H = E_n \bra{\psi_L^{(n)}},
$$
with the normalization
$$
\langle \psi_L^{(m)} | \psi_R^{(n)} \rangle = \delta_{mn}.
$$

We then compute all eigenvalues $\{E_n\}$ and left/right eigenvectors $\{\ket{\psi_R^{(n)}}, \bra{\psi_L^{(n)}}\}$ once, normalize them, and store them.

To identify **spectral loops**, the eigenvalues are clustered by radius $|E_n|$ using KMeans with a fixed $k = 3$. The two large clusters correspond to the **outer** and **inner** spectral loops in the complex plane, while the smallest cluster is interpreted as **isolated (topological) modes** and is excluded from the loop–cycle correspondence analysis.

Let $\ell$ be one such loop label and define
$$
\mathcal{I}_\ell = \{ n \mid \text{cluster label of } E_n \text{ is } \ell \},
\qquad
\Gamma_\ell = \{ E_n : n \in \mathcal{I}_\ell \}.
$$

The associated **biorthogonal projector** onto that loop is
$$
P_\ell 
= \sum_{n \in \mathcal{I}_\ell} 
  \ket{\psi_R^{(n)}} \bra{\psi_L^{(n)}}.
$$

The goal here is to find how the inverse image of the **spectral loop** $\Gamma_\ell$ can reveal a **lattice cycle** $C_\ell$ in real space.
"""
        )


def _section_biorthogonal_weights() -> None:
    st.markdown("### Ⅱ. Biorthogonal Node / Edge Weights")
    with st.expander("Click here to expand", expanded=False):
        st.markdown(
            r"""
Work in the site basis $\{ \ket{i} \}_{i=1}^N$. For a given spectral loop $\ell$ we define **node** and **edge** weights as biorthogonal expectation values of the projector $P_\ell$.

#### 1. Node weights

For each site $i$,
$$
w_i^{(\ell)} 
= \sum_{n \in \mathcal{I}_\ell}
  \bigl| \psi_{L,i}^{(n)} \bigr|
  \bigl| \psi_{R,i}^{(n)} \bigr|,
$$
where $\psi_{R,i}^{(n)} = \langle i | \psi_R^{(n)}\rangle$ and 
$\psi_{L,i}^{(n)} = \langle \psi_L^{(n)} | i\rangle$.

These are the **node biorthogonal weights**: they tell you how strongly the projector onto loop $\ell$ is supported on each site. In the code they are computed by summing $|V_L| \times |V_R|$ over the eigenmodes in the chosen cluster.

#### 2. Edge weights

For each directed edge $i \to j$ with matrix element $H_{ij}$, we define the **directed edge biorthogonal weight**
$$
W_{i\to j}^{(\ell)} 
= |H_{ij}| 
  \sum_{n \in \mathcal{I}_\ell}
    \bigl|\psi_{L,i}^{(n)}\bigr|
    \bigl|\psi_{R,j}^{(n)}\bigr|.
$$

We then build an undirected graph $G$ from the nonzero entries of $H$, and for each undirected edge $\{i,j\}$ we use the symmetrized weight
$$
W_{\{i,j\}}^{(\ell)} 
= W_{i\to j}^{(\ell)} + W_{j\to i}^{(\ell)}.
$$

These are the **edge biorthogonal weights** used in the visualizations and in the NMI metric.

The *Lattice metric* switch in the sidebar chooses how these weights are shown:

- **`node_centric`:** nodes are **sized and colored** by $w_i^{(\ell)}$;
- **`edge_centric`:** edges are **thickened and colored** by $W_{\{i,j\}}^{(\ell)}$.
"""
        )


def _section_cycles_and_nmi() -> None:
    st.markdown("### Ⅲ. Real-Space Cycles and NMI")
    with st.expander("Click here to expand", expanded=False):
        st.markdown(
            r"""
### Real-space cycles

We consider the undirected lattice graph $G$ (built from the nonzero entries of $H$) and enumerate simple cycles (closed paths with no repeated nodes) of a chosen length $L$:
$$
C = (i_1, i_2, \dots, i_L), \qquad
(i_k, i_{k+1}) \in E(G), \quad i_{L+1} \equiv i_1.
$$

The code identifies cycles that are equivalent up to cyclic permutations and reversal (so each undirected cycle is only counted once).

For each undirected edge $e$ of $G$ and cycle $C$, define the binary indicator
$$
X_e(C) =
\begin{cases}
1, & e \in C,\\
0, & e \notin C,
\end{cases}
$$
and the **total spectral weight on the cycle**
$$
W_{\text{cycle}}^{(\ell)}(C) 
= \sum_{e \in C} W_e^{(\ell)}.
$$

$W_{\text{cycle}}^{(\ell)}$ is reported in the table as `total_weight`.

---

### Mutual information and normalized MI (NMI)

To **quantify** how well a given cycle $C$ captures the spatial structure of the loop projector $P_\ell$, we compare:

- $X(C) = \{ X_e(C) \}_{e \in E(G)}$: a binary pattern indicating which edges lie on the cycle;
- $Y = \{ W_e^{(\ell)} \}_{e \in E(G)}$: the continuous spectral weights on edges.

We treat $(X_e(C), Y_e)$ over all edges $e$ as samples of two random variables $X$ (binary) and $Y$ (non-negative) and estimate the **mutual information**
$$
I_\ell(C) 
= I\big(X(C); Y\big) \ge 0
$$
using a non-parametric estimator (calling `scikit-learn`'s `mutual_info_regression` with $X$ marked as discrete).

To obtain a bounded, correlation-like score, we map the mutual information to a **normalized MI**
$$
\mathrm{NMI}_\ell(C)
= \sqrt{\,1 - e^{-2 I_\ell(C)}\,} \in [0, 1).
$$

This is the NMI value shown (e.g. in the sidebar note
$\mathrm{NMI} = \sqrt{1 - e^{-2\,\mathrm{MI}}}$).

The inspiration for this transformation is that it inverts the standard formula relating mutual information to the correlation coefficient in a bivariate normal model.

For a bivariate normal model with correlation coefficient $\rho$, the mutual information is:
$$
I(X;Y) = -\tfrac{1}{2}\log(1-\rho^2),
$$
so the above transformation would recover $|\rho|$. In that sense, $\mathrm{NMI}_\ell(C)$ can be interpreted as a novel “correlation coefficient” between spectral loop and lattice cycle.

We compute $\mathrm{NMI}_\ell(C)$ for **every** simple cycle of length $L$, sort cycles in descending order of NMI, and let you browse cycles by **NMI rank**.
"""
        )


def _section_how_to_read_dashboard() -> None:
    st.markdown("### Ⅳ. How to Read the Dashboard")
    with st.expander("Click here to expand", expanded=False):
        st.markdown(
            r"""
1. **Top row – spectrum:**  
   Complex eigenvalues $E_n$ in the plane. Points are colored by KMeans cluster (spectral loop). The smallest cluster (fewest eigenvalues) is drawn as “Isolated modes” and ignored in the loop–cycle analysis. The selected loop is highlighted.

2. **Bottom-left – real-space lattice:**  
   The lattice graph in real space. Depending on the *Lattice metric*:
   - `node_centric`: node size / color = $w_i^{(\ell)}$;
   - `edge_centric`: edge thickness / color = $W_{\{i,j\}}^{(\ell)}$.

3. **Bottom-right – top NMI cycle:**  
   The same lattice, but with a single simple cycle highlighted:
   $$
   C_{\text{sel}} = \arg\max_C \mathrm{NMI}_\ell(C)
   $$
   within the chosen cycle length $L$ and NMI rank. This is the candidate **real-space cycle** associated with the chosen spectral loop.

Below the figure, the app summarizes (for the selected loop and cycle):

- cluster label and number of eigenvalues in the loop;
- NMI rank and cycle length $L$;
- NMI score and total edge weight on the cycle.
"""
        )


def _section_controls_and_observations() -> None:
    st.markdown("### Ⅴ. Controls and Dataset-Specific Observations")
    with st.expander("Click here to expand", expanded=False):
        st.markdown(
            r"""
### Controls and typical usage

- **Target spectral loop**  
  Choose which KMeans cluster (loop) to analyze. The smallest-count cluster (isolated modes) is automatically ignored in the NMI search.

- **Lattice metric**  
  Switch between `node_centric` or `edge_centric` visualization of biorthogonal weights.

- **Layout algorithm**  
  Either a Kamada–Kawai layout or a planar layout (fallback to Kamada–Kawai if the graph is not planar).

- **Cycle length $L$**  
  - $L = 0$: use the **default length**, equal to the number of eigenvalues in the chosen spectral loop (i.e. $|\mathcal{I}_\ell|$).
    - **This is useful for the desire to search a “Hatano–Nelson-like” correspondence where the spectral loop length matches the physical cycle length.**
  - $L > 0$: enumerate cycles of exactly that length.  

  For this Hamiltonian, cycles are precomputed for $L$ from the default length up to $L_{\max} = 88$.\
  **This is useful to visualize the correspondence between the inner spectral loop and its weight-concentrated lattice cycle, which has length $L = 88$.**

- **NMI cycle rank**  
  Browse cycles sorted by NMI (rank $1$ is the highest NMI). Rank only appears when there is more than one simple cycle of the chosen length.

---

### Dataset-specific observations (for the current Hamiltonian)

- **Outer spectral loop.**  
  If you choose the outer loop and use $L = 0$ (default, e.g. $L = 32$), you will see a real-space cycle with a strong concentration of biorthogonal weights. In the NMI ranking, two cycles (ranks 1 and 2) can have identical NMI due to lattice symmetries; one of them (e.g. rank 2) coincides with the visually dominant biorthogonal-weight cycle. The other is a symmetry-related variant; however, its origin is unclear — it may correspond to a different weighting mechanism.

- **Inner spectral loop.**  
  For the inner loop, the biorthogonal-weight concentration cycle is **longer** than the spectral loop, and has length $L = 88$. If you keep $L = 0$ / $L = 32$, the top NMI cycles are short and asymmetric. Setting $L = 88$ produces a symmetric cycle whose NMI-optimal representative (rank 1) matches the visually dominant biorthogonal-weight cycle.

This asymmetry between spectral-loop length and physical cycle length is one of the key non-trivial features of this model.
"""
        )


def section_conceptual_diagrams() -> None:
    with st.expander("Ⅵ. Conceptual Diagrams", expanded=True):
        st.markdown("**Overall pipeline**")
        st_mermaid(
            r"""
graph TD
    H(Hamiltonian H) --> Evals(Eigenvalues E_n)
    Evals --> Clusters[Cluster into \n spectral loops Γ_ℓ]
    Clusters --> LoopPlot{{Plot: Spectral loops}}
    Clusters --> Projector[Loop projector P_ℓ]
    Projector --> Weights[Biorthogonal \n node/edge weights]
    Weights --> WeightPlot{{Plot: biorthogonal weights}}
    Weights --> Cycles[Enumerate lattice cycles C \n of length L]
    Cycles --> MI["Compute 'atomic' \n mutual information \n I(X(C); Y)"]
    MI --> NMI["Rank cycles by NMI_ℓ(C)"]
    NMI --> MIPlot{{Plot: top NMI cycle C_topk}}
    LoopPlot -. "Physical Correspondence" .-> WeightPlot
    WeightPlot <-. "Computational Correspondence \nand Visual Match" .-> MIPlot
    linkStyle 10 stroke:#8b0000, stroke-width:3px
    linkStyle 11 stroke:#8b0000, stroke-width:3px 
            """
        )

        st.markdown("**Loop–cycle duality for this Hamiltonian**")
        st_mermaid(
            r"""
graph LR
    subgraph Spectrum
        O["Outer spectral loop Γ_1"]
        I["Inner spectral loop Γ_2"]
    end
    subgraph Lattice
        Co[("Inner lattice cycle C_1<br>(L=32 — same as Γ_1)")]
        Ci[("Outer lattice cycle C_2<br>(L=88 — longer than Γ_2)")]
    end
    O <-.-> Co
    I <-.-> Ci
    style Co stroke-width:3px
    style Ci stroke-width:3px
            """
        )


def render_theory_intro() -> None:
    """Render the theory intro, partitioned into multiple expandable sections.

    Some key sections (Ⅰ–Ⅱ) are expanded by default; others are collapsed
    to avoid overwhelming the reader. Mermaid diagrams are rendered with
    ``st_mermaid`` instead of fenced code blocks.
    """
    _section_physical_context_and_method()
    _section_biorthogonal_weights()
    _section_cycles_and_nmi()
    _section_how_to_read_dashboard()
    _section_controls_and_observations()