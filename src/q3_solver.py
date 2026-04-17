from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Sequence

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB
from sklearn.cluster import SpectralClustering

from data_pipeline import DataValidationError, LogisticsInstance, TemporalCompatibilityGraph


@dataclass(frozen=True, slots=True)
class ClusterQuality:
    """Partition quality metrics: modularity and mean intra-cluster demand variance."""

    modularity: float
    demand_variance: float


@dataclass(frozen=True, slots=True)
class ClusterPartition:
    """Spectral partition of the customer compatibility graph."""

    labels: np.ndarray
    clusters: Dict[int, List[int]]
    affinity_matrix: np.ndarray
    quality: ClusterQuality


@dataclass(frozen=True, slots=True)
class RouteEvaluation:
    """Classical no-wait rollout of a pure spatial route."""

    route: List[int]
    arrival_times: Dict[int, float]
    penalties: Dict[int, float]
    total_penalty: float
    travel_cost: float
    objective: float
    finish_time: float


@dataclass(frozen=True, slots=True)
class ClusterSolveResult:
    """Local cluster solution after asynchronous weight updates."""

    cluster_id: int
    route: List[int]
    arrival_times: Dict[int, float]
    penalties: Dict[int, float]
    total_penalty: float
    travel_cost: float
    objective: float
    finish_time: float
    iterations: int
    weight_matrix: np.ndarray

    @property
    def entry_node(self) -> int:
        return self.route[0]

    @property
    def exit_node(self) -> int:
        return self.route[-1]


@dataclass(frozen=True, slots=True)
class GlobalRouteResult:
    """Final stitched single-vehicle route over all customers."""

    supernode_order: List[int]
    route: List[int]
    arrival_times: Dict[int, float]
    penalties: Dict[int, float]
    total_penalty: float
    travel_cost: float
    objective: float


@dataclass(frozen=True, slots=True)
class Q3Solution:
    """Full two-stage problem-3 solution bundle."""

    partition: ClusterPartition
    cluster_results: List[ClusterSolveResult]
    global_result: GlobalRouteResult


class GraphClusterer:
    """Physical graph reducer for the 50-customer network.

    The directed compatibility matrix \(W\) is symmetrized as
    \(A=0.5(W+W^\top)\), truncated below at zero, and normalized by setting the
    diagonal to one. The resulting undirected affinity supports spectral
    clustering and modularity evaluation.
    """

    def __init__(self, instance: LogisticsInstance, compatibility_graph: TemporalCompatibilityGraph, n_clusters: int = 5, random_state: int = 42) -> None:
        self._instance = instance
        self._compatibility_graph = compatibility_graph
        self._n_clusters = n_clusters
        self._random_state = random_state
        if n_clusters <= 0:
            raise DataValidationError("n_clusters must be positive.")

    def build_affinity_matrix(self) -> np.ndarray:
        matrix = self._compatibility_graph.build_compatibility_matrix()
        affinity = 0.5 * (matrix + matrix.T)
        affinity = np.maximum(affinity, 0.0)
        np.fill_diagonal(affinity, 1.0)
        return affinity

    def cluster(self) -> ClusterPartition:
        affinity = self.build_affinity_matrix()
        labels = SpectralClustering(n_clusters=self._n_clusters, affinity="precomputed", random_state=self._random_state, assign_labels="kmeans").fit_predict(affinity)
        clusters: Dict[int, List[int]] = {k: [] for k in range(self._n_clusters)}
        for idx, customer_id in enumerate(self._instance.customer_ids):
            clusters[int(labels[idx])].append(customer_id)
        clusters = {k: sorted(v) for k, v in clusters.items() if v}
        demands = np.asarray([node.demand for node in self._instance.customers], dtype=np.float64)
        quality = self.evaluate_quality(affinity, labels, demands)
        return ClusterPartition(labels, clusters, affinity, quality)

    @staticmethod
    def evaluate_quality(affinity_matrix: np.ndarray, labels: np.ndarray, demands: np.ndarray) -> ClusterQuality:
        """Compute weighted modularity and mean intra-cluster demand variance."""

        graph = nx.Graph()
        graph.add_nodes_from(range(affinity_matrix.shape[0]))
        for i in range(affinity_matrix.shape[0]):
            for j in range(i + 1, affinity_matrix.shape[1]):
                weight = float(affinity_matrix[i, j])
                if weight > 0.0:
                    graph.add_edge(i, j, weight=weight)
        communities = [set(np.where(labels == label)[0]) for label in np.unique(labels)]
        modularity = 0.0 if graph.number_of_edges() == 0 else float(nx.community.modularity(graph, communities, weight="weight"))
        variances = [float(np.var(demands[list(nodes)])) for nodes in communities if nodes]
        return ClusterQuality(modularity, float(np.mean(variances)) if variances else 0.0)


class AsyncClusterSolver:
    """Local master-slave iterative solver using only edge-space decisions.

    Spatial order is optimized on a weighted complete digraph. Time is rolled out
    classically under the no-wait rule. Node penalties are pushed back onto the
    incoming route edges, then learned through the EMA update
    \(w^{k+1}_{ij}=t_{ij}+\max(0,0.9(w^k_{ij}-t_{ij})+k^{-1/2}\Delta w_{ij})\).
    """

    def __init__(self, instance: LogisticsInstance, cluster_node_ids: Sequence[int], max_iterations: int = 20, tolerance: float = 1e-3) -> None:
        self._instance = instance
        self._cluster_node_ids = list(cluster_node_ids)
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        if not self._cluster_node_ids:
            raise DataValidationError("cluster_node_ids cannot be empty.")
        if len(self._cluster_node_ids) > 15:
            raise DataValidationError("Cluster size must not exceed 15.")

    def solve(self, cluster_id: int) -> ClusterSolveResult:
        base = self._submatrix()
        current = base.copy()
        best_eval: RouteEvaluation | None = None
        best_weights = current.copy()
        last_penalty: float | None = None
        used_iterations = 0
        for k in range(1, self._max_iterations + 1):
            used_iterations = k
            route = self._mock_quantum_tsp_solver(current)
            evaluation = self._evaluate_time_penalties(route)
            if best_eval is None or evaluation.objective < best_eval.objective:
                best_eval = evaluation
                best_weights = current.copy()
            if last_penalty is not None and abs(evaluation.total_penalty - last_penalty) <= self._tolerance:
                break
            delta = self._delta_weights(route, evaluation.penalties)
            learned = np.maximum(0.0, 0.9 * (current - base) + delta / sqrt(k))
            current = base + learned
            last_penalty = evaluation.total_penalty
        if best_eval is None:
            raise RuntimeError("Failed to obtain a cluster route.")
        return ClusterSolveResult(cluster_id, best_eval.route, best_eval.arrival_times, best_eval.penalties, best_eval.total_penalty, best_eval.travel_cost, best_eval.objective, best_eval.finish_time, used_iterations, best_weights)

    def _submatrix(self) -> np.ndarray:
        idx = np.asarray(self._cluster_node_ids, dtype=int)
        matrix = self._instance.travel_time_matrix[np.ix_(idx, idx)].astype(np.float64)
        return np.where(np.isfinite(matrix), matrix, 1e6)

    def _mock_quantum_tsp_solver(self, weight_matrix: np.ndarray) -> List[int]:
        """Exact pure-space Hamiltonian path solver on the cluster graph."""

        n = len(self._cluster_node_ids)
        if n == 1:
            return [self._cluster_node_ids[0]]
        full = (1 << n) - 1
        dp = np.full((1 << n, n), np.inf, dtype=np.float64)
        parent = np.full((1 << n, n), -1, dtype=int)
        for j in range(n):
            dp[1 << j, j] = 0.0
        for mask in range(1 << n):
            for j in range(n):
                if not (mask & (1 << j)) or not np.isfinite(dp[mask, j]):
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    cand = dp[mask, j] + weight_matrix[j, nxt]
                    if cand < dp[new_mask, nxt]:
                        dp[new_mask, nxt] = cand
                        parent[new_mask, nxt] = j
        last = int(np.argmin(dp[full]))
        order: List[int] = []
        mask = full
        while last != -1:
            order.append(self._cluster_node_ids[last])
            prev = int(parent[mask, last])
            mask ^= 1 << last
            last = prev
        order.reverse()
        return order

    def _evaluate_time_penalties(self, route: Sequence[int], start_time: float | None = None) -> RouteEvaluation:
        """Roll out a route with no waiting and quadratic early/late penalties."""

        if not route:
            raise DataValidationError("route cannot be empty.")
        arrivals: Dict[int, float] = {}
        penalties: Dict[int, float] = {}
        first = self._instance.get_node(route[0])
        current_time = first.ready_time if start_time is None else start_time
        travel_cost = 0.0
        arrivals[first.id] = current_time
        penalties[first.id] = self._node_penalty(first.id, current_time)
        for prev_id, node_id in zip(route[:-1], route[1:]):
            prev = self._instance.get_node(prev_id)
            travel = float(self._instance.travel_time_matrix[prev_id, node_id])
            travel_cost += travel
            current_time += prev.service_time + travel
            arrivals[node_id] = current_time
            penalties[node_id] = self._node_penalty(node_id, current_time)
        finish_time = current_time + self._instance.get_node(route[-1]).service_time
        total_penalty = float(sum(penalties.values()))
        return RouteEvaluation(list(route), arrivals, penalties, total_penalty, travel_cost, travel_cost + total_penalty, finish_time)

    def _node_penalty(self, node_id: int, arrival_time: float) -> float:
        node = self._instance.get_node(node_id)
        early = max(node.ready_time - arrival_time, 0.0)
        late = max(arrival_time - node.due_time, 0.0)
        return 10.0 * early * early + 20.0 * late * late

    def _delta_weights(self, route: Sequence[int], penalties: Dict[int, float]) -> np.ndarray:
        delta = np.zeros((len(self._cluster_node_ids), len(self._cluster_node_ids)), dtype=np.float64)
        local_pos = {node_id: idx for idx, node_id in enumerate(self._cluster_node_ids)}
        for prev_id, node_id in zip(route[:-1], route[1:]):
            i = local_pos[prev_id]
            j = local_pos[node_id]
            delta[i, j] += penalties[node_id]
        return delta


class GlobalStitcher:
    """Supernode reconstructor for the cluster-level global order."""

    def __init__(self, instance: LogisticsInstance, cluster_results: Sequence[ClusterSolveResult]) -> None:
        self._instance = instance
        self._cluster_results = list(cluster_results)

    def stitch(self) -> GlobalRouteResult:
        matrix = self.build_supernode_cost_matrix()
        order = self._solve_supernode_tsp(matrix)
        by_id = {result.cluster_id: result for result in self._cluster_results}
        route: List[int] = []
        for cluster_id in order[1:]:
            route.extend(by_id[cluster_id].route)
        evaluation = self._evaluate_full_route(route)
        return GlobalRouteResult(order, route, evaluation.arrival_times, evaluation.penalties, evaluation.total_penalty, evaluation.travel_cost, evaluation.objective)

    def build_supernode_cost_matrix(self) -> np.ndarray:
        """Cost matrix on depot plus cluster supernodes using boundary-node transfers."""

        size = len(self._cluster_results) + 1
        matrix = np.full((size, size), 1e6, dtype=np.float64)
        for i in range(size):
            matrix[i, i] = 1e6
        for col, target in enumerate(self._cluster_results, start=1):
            entry_time = float(self._instance.travel_time_matrix[0, target.entry_node])
            matrix[0, col] = self._transition_cost(target.route, entry_time)
        for row, source in enumerate(self._cluster_results, start=1):
            matrix[row, 0] = float(self._instance.travel_time_matrix[source.exit_node, 0])
            for col, target in enumerate(self._cluster_results, start=1):
                if row == col:
                    continue
                jump = float(self._instance.travel_time_matrix[source.exit_node, target.entry_node])
                matrix[row, col] = jump + self._transition_cost(target.route, source.finish_time + jump)
        return matrix

    def _transition_cost(self, route: Sequence[int], first_arrival_time: float) -> float:
        return AsyncClusterSolver(self._instance, route)._evaluate_time_penalties(route, start_time=first_arrival_time).objective

    def _evaluate_full_route(self, route: Sequence[int]) -> RouteEvaluation:
        if not route:
            raise DataValidationError("Full route cannot be empty.")
        arrivals: Dict[int, float] = {}
        penalties: Dict[int, float] = {}
        first_travel = float(self._instance.travel_time_matrix[0, route[0]])
        current_time = first_travel
        travel_cost = first_travel
        arrivals[route[0]] = current_time
        penalties[route[0]] = self._node_penalty(route[0], current_time)
        for prev_id, node_id in zip(route[:-1], route[1:]):
            prev = self._instance.get_node(prev_id)
            travel = float(self._instance.travel_time_matrix[prev_id, node_id])
            travel_cost += travel
            current_time += prev.service_time + travel
            arrivals[node_id] = current_time
            penalties[node_id] = self._node_penalty(node_id, current_time)
        travel_cost += float(self._instance.travel_time_matrix[route[-1], 0])
        total_penalty = float(sum(penalties.values()))
        finish_time = current_time + self._instance.get_node(route[-1]).service_time
        return RouteEvaluation(list(route), arrivals, penalties, total_penalty, travel_cost, travel_cost + total_penalty, finish_time)

    def _node_penalty(self, node_id: int, arrival_time: float) -> float:
        node = self._instance.get_node(node_id)
        early = max(node.ready_time - arrival_time, 0.0)
        late = max(arrival_time - node.due_time, 0.0)
        return 10.0 * early * early + 20.0 * late * late

    def _solve_supernode_tsp(self, cost_matrix: np.ndarray) -> List[int]:
        nodes = list(range(cost_matrix.shape[0]))
        model = gp.Model("q3_supernode_tsp")
        model.Params.OutputFlag = 0
        x = model.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(cost_matrix[i, j] * x[i, j] for i, j in x.keys()), GRB.MINIMIZE)
        for i in nodes:
            model.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == 1)
            model.addConstr(gp.quicksum(x[j, i] for j in nodes if i != j) == 1)
        model.Params.LazyConstraints = 1

        def callback(cb_model: gp.Model, where: int) -> None:
            if where != GRB.Callback.MIPSOL:
                return
            values = cb_model.cbGetSolution(x)
            arcs = [(i, j) for (i, j), val in values.items() if val > 0.5]
            cycle = self._shortest_cycle(arcs, nodes)
            if len(cycle) < len(nodes):
                cb_model.cbLazy(gp.quicksum(x[i, j] for i in cycle for j in cycle if i != j) <= len(cycle) - 1)

        model.optimize(callback)
        successor = {i: j for (i, j) in x.keys() if x[i, j].X > 0.5}
        order = [0]
        while True:
            nxt = successor[order[-1]]
            if nxt == 0:
                break
            order.append(nxt)
        return order

    @staticmethod
    def _shortest_cycle(arcs: Sequence[tuple[int, int]], nodes: Sequence[int]) -> List[int]:
        successor = {i: j for i, j in arcs}
        unseen = set(nodes)
        best = list(nodes)
        while unseen:
            start = unseen.pop()
            cycle = [start]
            nxt = successor[start]
            while nxt != start:
                cycle.append(nxt)
                unseen.discard(nxt)
                nxt = successor[nxt]
            if len(cycle) < len(best):
                best = cycle
        return best


class Q3Solver:
    """High-level orchestrator for the two-stage problem-3 pipeline."""

    def __init__(self, instance: LogisticsInstance, compatibility_graph: TemporalCompatibilityGraph, n_clusters: int = 5, local_max_iterations: int = 20) -> None:
        self._instance = instance
        self._compatibility_graph = compatibility_graph
        self._n_clusters = n_clusters
        self._local_max_iterations = local_max_iterations

    def solve(self) -> Q3Solution:
        partition = GraphClusterer(self._instance, self._compatibility_graph, self._n_clusters).cluster()
        cluster_results = [AsyncClusterSolver(self._instance, node_ids, self._local_max_iterations).solve(cluster_id) for cluster_id, node_ids in sorted(partition.clusters.items())]
        global_result = GlobalStitcher(self._instance, cluster_results).stitch()
        return Q3Solution(partition, cluster_results, global_result)


__all__: List[str] = ["AsyncClusterSolver", "ClusterPartition", "ClusterQuality", "ClusterSolveResult", "GlobalRouteResult", "GlobalStitcher", "GraphClusterer", "Q3Solution", "Q3Solver", "RouteEvaluation"]
