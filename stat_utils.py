from typing import Dict, Tuple, Any
from numpy.typing import NDArray

import json
from collections import Counter

import numpy as np
import scipy.sparse as sp
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
    result = dict(zip(nodes, membership))

    last_community_sizes_dict = dict(Counter(result.values()))
    last_community_count = len(last_community_sizes_dict)

    last_community_sizes = json.dumps(last_community_sizes_dict)

    return last_community_count, last_community_sizes


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


def compute_all_stats(record: RawSimulationRecord) -> Dict[str, Any]:
    """Compute all analysis metrics for a loaded simulation record.

    Returns a dict suitable for msgpack serialisation with fields:
      convergence_step      int   – step at which opinions converged
      log_convergence_time  float – log1p(convergence_step)
      final_variance        float – variance of opinions at the final step
      final_magnetization   float – |mean opinion| at the final step
      n_closed_triangles    int   – closed directed triangles in the final graph
      community_count       int   – number of Leiden communities
      community_sizes       str   – JSON {community_id: size}
    """
    conv_step = get_convergence_step(record)
    variance, magnetization = get_opinion_stats(record)

    last_graph = record.get_graph(record.max_step)
    nodelist = sorted(last_graph.nodes())
    A = nx.to_numpy_array(last_graph, nodelist=nodelist)
    n_triads, _ = get_triads_stats(A)

    # Reuse the already-loaded graph instead of calling record.get_graph a second time.
    community_count, community_sizes = get_last_community_count(last_graph)

    return {
        "convergence_step": conv_step,
        "log_convergence_time": float(np.log1p(conv_step)),
        "final_variance": variance,
        "final_magnetization": magnetization,
        "n_closed_triangles": int(n_triads),
        "community_count": community_count,
        "community_sizes": community_sizes,
    }
