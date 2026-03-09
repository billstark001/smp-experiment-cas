from typing import Dict, Tuple, Any, Optional
from numpy.typing import NDArray

import json
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.signal import find_peaks
from scipy.stats import norm, gaussian_kde
import igraph as ig
import leidenalg
import networkx as nx


from smp_bindings import RawSimulationRecord  # type: ignore


def get_triads_stats(A: NDArray):
    # counts the number of A->B, B->C, A->C triads in the graph represented by adjacency matrix A
    # Use sparse matrix multiplication to avoid O(N³) dense matmul — critical for large graphs.
    S = sp.csr_matrix(A)
    S2 = S @ S  # sparse: 2-path counts
    # closed triads: positions where both A[i,j] and A²[i,j] are nonzero
    S_triads = S.multiply(S2)
    n_triads = int(S_triads.sum())

    return n_triads, S_triads.toarray()


def get_last_community_count(last_graph: nx.DiGraph):
    # Build a 0-indexed igraph from the networkx graph.
    # Mapping is required because igraph add_edges() uses integer vertex *indices*,
    # not the original node labels — passing raw labels causes wrong or out-of-range edges.
    nodes = list(last_graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edges_indexed = [(node_to_idx[u], node_to_idx[v]) for u, v in last_graph.edges()]

    igraph_g = ig.Graph(n=len(nodes), edges=edges_indexed, directed=True)

    # n_iterations > 0 bounds the refinement passes and prevents potential runaway.
    partition = leidenalg.find_partition(
        igraph_g,
        leidenalg.ModularityVertexPartition,
        n_iterations=50,
    )

    membership = partition.membership
    modularity = partition.quality()
    result = dict(zip(nodes, membership))

    last_community_sizes_dict = dict(Counter(result.values()))
    last_community_count = len(last_community_sizes_dict)

    last_community_sizes = json.dumps(last_community_sizes_dict)

    return last_community_count, last_community_sizes, modularity


def kde_min_bw_factory(min_bandwidth: float):
    def min_bw_factor(kde_obj):
        default_factor = kde_obj.scotts_factor()
        min_factor = min_bandwidth / np.std(kde_obj.dataset, ddof=1)
        return max(default_factor, min_factor)

    return min_bw_factor


def get_kde_pdf(data, min_bandwidth: float, xmin: float, xmax: float):
    # If all samples are identical, KDE will raise an error; handle this as a special case.
    if np.all(data == data[0]):
        # Return an approximate delta distribution with a peak at that point.
        def delta_like_pdf(x):
            return norm.pdf(x, data[0], min_bandwidth)
            # return 1.0 if np.isclose(x[0], data[0]) else 0.0

        return delta_like_pdf

    bw_method = kde_min_bw_factory(min_bandwidth)
    data_smpl = data  # np.random.choice(data, size=2000, replace=False)
    kde = gaussian_kde(data_smpl, bw_method=bw_method)
    return kde


def get_last_opinion_peak_count(opinion_last: NDArray, opinion_peak_distance: int):
    """Compute the number of opinion peaks in the final opinion distribution.

    Args:
        opinion_last (NDArray): The opinions at the final simulation step.
        opinion_peak_distance (int): Minimum distance between peaks.

    Returns:
        int: Number of peaks in the final opinion distribution.
    """
    kde = get_kde_pdf(opinion_last, 0.1, -1, 1)

    x_grid = np.linspace(-1, 1, 1001)
    y_kde = kde(x_grid)

    height_threshold = np.max(y_kde) * 0.1
    peaks, _ = find_peaks(
        y_kde, height=height_threshold, distance=opinion_peak_distance
    )

    return len(peaks)


def get_opinion_stats(record: RawSimulationRecord) -> Tuple[float, float]:
    """Compute final opinion variance and magnetization.

    Returns:
      (variance, magnetization) where variance is the variance of the opinion
      distribution at the final simulation step, and magnetization is the
      absolute mean opinion — an order parameter that captures global consensus.
    """
    final_opinions = record.opinions[-1].astype(np.float64)
    variance = float(np.var(final_opinions))
    magnetization = float(np.abs(np.mean(final_opinions)))
    return variance, magnetization


def get_convergence_step(record: RawSimulationRecord, epsilon: float = 1e-4) -> int:
    """Return the step at which opinions effectively converge.

    Convergence is defined as the last step where the maximum per-agent opinion
    change exceeds *epsilon*.  Returns 0 if opinions never exceeded that
    threshold, or record.max_step if convergence was never reached.
    """
    opinions = record.opinions.astype(np.float64)  # shape: (steps+1, agents)
    diffs = np.abs(np.diff(opinions, axis=0))  # shape: (steps, agents)
    max_change = diffs.max(axis=1)  # shape: (steps,)
    changing = np.where(max_change > epsilon)[0]
    if len(changing) == 0:
        return 0
    # diff[i] = opinions[i+1] - opinions[i], so the last active step is changing[-1]+1
    return int(changing[-1]) + 1


def compute_all_stats(
    record: RawSimulationRecord,
    pre_computed: Optional[Dict[str, Any]] = None,
    opinion_peak_distance: int = 50,
) -> Dict[str, Any]:
    """Compute all analysis metrics for a loaded simulation record.

    Args:
        record: The simulation record to analyse.
        pre_computed: Optional dict of already-computed fields; any key present
            here is skipped and its value is carried over verbatim.
        opinion_peak_distance: Minimum sample distance between KDE peaks used
            for opinion_peak_count (default 50).

    Returns a dict suitable for msgpack serialization with fields:
        convergence_step      int   – step at which opinions converged
        log_convergence_time  float – log1p(convergence_step)
        final_variance        float – variance of opinions at the final step
        final_magnetization   float – |mean opinion| at the final step
        opinion_peak_count    int   – number of KDE peaks in the final opinion distribution
        n_closed_triangles    int   – closed directed triangles in the final graph
        community_count       int   – number of Leiden communities
        community_sizes       str   – JSON {community_id: size}
        modularity            float – Leiden modularity of the final graph
    """
    result: Dict[str, Any] = dict(pre_computed) if pre_computed else {}

    if "convergence_step" not in result:
        result["convergence_step"] = get_convergence_step(record)

    if "log_convergence_time" not in result:
        result["log_convergence_time"] = float(np.log1p(result["convergence_step"]))

    need_opinions = {
        "final_variance",
        "final_magnetization",
        "opinion_peak_count",
    } - result.keys()
    if need_opinions:
        final_opinions = record.opinions[-1].astype(np.float64)
        result["final_variance"] = float(np.var(final_opinions))
        result["final_magnetization"] = float(np.abs(np.mean(final_opinions)))
        result["opinion_peak_count"] = get_last_opinion_peak_count(
            final_opinions, opinion_peak_distance
        )

    need_graph = {
        "n_closed_triangles",
        "community_count",
        "community_sizes",
        "modularity",
    } - result.keys()
    if need_graph:
        last_graph = record.get_graph(record.max_step)
        if "n_closed_triangles" not in result:
            nodelist = sorted(last_graph.nodes())
            A = nx.to_numpy_array(last_graph, nodelist=nodelist)
            n_triads, _ = get_triads_stats(A)
            result["n_closed_triangles"] = int(n_triads)
        if {"community_count", "community_sizes", "modularity"} - result.keys():
            community_count, community_sizes, modularity = get_last_community_count(
                last_graph
            )
            result["community_count"] = community_count
            result["community_sizes"] = community_sizes
            result["modularity"] = modularity

    return result
