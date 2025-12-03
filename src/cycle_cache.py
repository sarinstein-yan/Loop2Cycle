import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from DualCycleAnalyzer import DualCycleAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "assets"
HAM_PATH = DATA_DIR / "Ham_96.csv"
CACHE_PATH = DATA_DIR / "nmi_cache_k3.pkl"

N_CLUSTERS = 3
MAX_CYCLE_LENGTH = 88


def compute_nmi_cache() -> Dict[str, Any]:
    if not HAM_PATH.exists():
        raise FileNotFoundError(f"Hamiltonian CSV not found at {HAM_PATH}")

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading Hamiltonian from {HAM_PATH} ...")
    df = pd.read_csv(HAM_PATH, header=None)
    H = df.values
    print(f"  Hamiltonian shape: {H.shape}")

    analyzer = DualCycleAnalyzer(H)

    # Cluster spectrum with fixed k
    analyzer.cluster_spectrum(N_CLUSTERS)
    labels = np.array(analyzer.labels)
    uniq, counts = np.unique(labels, return_counts=True)

    # Identify the smallest cluster (isolated eigenvalues) and ignore it
    ignored_cluster = int(uniq[np.argmin(counts)])
    loop_labels = [int(l) for l in uniq if l != ignored_cluster]

    print(f"  KMeans k={N_CLUSTERS}")
    print("  Cluster sizes: " + ", ".join(f"{int(l)}: {int(c)}" for l, c in zip(uniq, counts)))
    print(f"  Ignoring smallest cluster: {ignored_cluster}")

    # Build list of (loop_label, L) tasks
    from typing import List as _List, Tuple as _Tuple

    tasks: _List[_Tuple[int, int]] = []
    for lbl in loop_labels:
        default_L = analyzer._default_cycle_length(lbl)
        L_start = int(default_L)
        if L_start < 1:
            L_start = 1
        print(f"Loop {lbl}: default L = {L_start}, precomputing up to {MAX_CYCLE_LENGTH}")
        for L in range(L_start, MAX_CYCLE_LENGTH + 1):
            tasks.append((lbl, L))

    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore

    try:
        from joblib import Parallel, delayed  # type: ignore
        from joblib import parallel as joblib_parallel  # type: ignore
        have_joblib = True
    except Exception:  # pragma: no cover
        Parallel = delayed = joblib_parallel = None  # type: ignore
        have_joblib = False

    if tasks:
        if have_joblib:
            if tqdm is not None:
                from contextlib import contextmanager

                @contextmanager
                def tqdm_joblib(tqdm_object):
                    """Context manager to patch joblib to report into tqdm."""
                    class TqdmBatchCompletionCallback(  # type: ignore[attr-defined]
                        joblib_parallel.BatchCompletionCallBack
                    ):
                        def __call__(self, *args, **kwargs):
                            tqdm_object.update(n=self.batch_size)
                            return super().__call__(*args, **kwargs)

                    old_cb = joblib_parallel.BatchCompletionCallBack  # type: ignore[attr-defined]
                    joblib_parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback  # type: ignore[attr-defined]
                    try:
                        yield tqdm_object
                    finally:
                        joblib_parallel.BatchCompletionCallBack = old_cb  # type: ignore[attr-defined]
                        tqdm_object.close()

                print(
                    f"Computing NMI cycles for {len(tasks)} (loop, L) pairs using "
                    "joblib.Parallel (threads) ..."
                )
                with tqdm_joblib(tqdm(total=len(tasks), desc="Caching NMI cycles")):
                    Parallel(n_jobs=-1, prefer="threads")(
                        delayed(analyzer._get_cached_mi_results)(lbl, L)
                        for (lbl, L) in tasks
                    )
            else:
                print(
                    f"Computing NMI cycles for {len(tasks)} (loop, L) pairs using "
                    "joblib.Parallel (threads) ..."
                )
                Parallel(n_jobs=-1, prefer="threads")(
                    delayed(analyzer._get_cached_mi_results)(lbl, L)
                    for (lbl, L) in tasks
                )
        else:
            if tqdm is not None:
                print(
                    f"Computing NMI cycles for {len(tasks)} (loop, L) pairs "
                    "sequentially ..."
                )
                for lbl, L in tqdm(tasks, desc="Caching NMI cycles"):
                    analyzer._get_cached_mi_results(lbl, L)
            else:
                for lbl, L in tasks:
                    analyzer._get_cached_mi_results(lbl, L)

    ham_digest = hashlib.sha256(H.tobytes()).hexdigest()
    payload: Dict[str, Any] = {
        "ham_digest": ham_digest,
        "n_clusters": N_CLUSTERS,
        "labels": analyzer.labels,
        "mi_cache": analyzer._mi_cache,
        "ignored_cluster": ignored_cluster,
        "max_cycle_length": MAX_CYCLE_LENGTH,
    }
    return payload


def main() -> None:
    payload = compute_nmi_cache()
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved NMI cache for k={payload['n_clusters']} to {CACHE_PATH}")
    print(
        f"Ignored cluster: {payload['ignored_cluster']}, "
        f"max cycle length: {payload['max_cycle_length']}"
    )


if __name__ == "__main__":
    main()