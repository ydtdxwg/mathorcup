from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from data_pipeline import DataValidationError, LogisticsInstance, TemporalCompatibilityGraph
from q3_solver import AsyncClusterSolver


BIG_M_PENALTY: float = 1_000_000.0
VEHICLE_FIXED_COST: float = 10_000.0


@dataclass(frozen=True, slots=True)
class RouteColumn:
    """One route column in the set-partitioning master problem.

    The reduced cost used by column generation is
    \(\bar c_r = c_r - \sum_i \pi_i a_{ir}\), where `customers` defines the
    incidence vector \(a_{ir}\), and `cost` is the real route cost including depot
    travel plus soft time-window penalties.
    """

    route: List[int]
    customers: List[int]
    cost: float
    load: float


@dataclass(frozen=True, slots=True)
class PricingResult:
    """Output of one pricing iteration."""

    selected_nodes: List[int]
    column: RouteColumn | None
    reduced_cost: float


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Final root-node price-and-branch result for one `V_max`."""

    objective_lp: float
    objective_ip: float | None
    columns: List[RouteColumn]
    lambda_lp: Dict[int, float]
    lambda_ip: Dict[int, float] | None


class MasterProblem:
    """Restricted master problem for CVRP-TW set covering.

    The LP relaxation is
    \[\min \sum_{r\in\Omega} c_r\lambda_r\]
    subject to customer covering and a vehicle-count cap
    \[\sum_{r\in\Omega} a_{ir}\lambda_r \ge 1,\quad \sum_{r\in\Omega}\lambda_r\le V_{max},\quad \lambda_r\ge 0.\]
    """

    def __init__(self, instance: LogisticsInstance, columns: Sequence[RouteColumn], v_max: int) -> None:
        self._instance = instance
        self._columns = list(columns)
        self._v_max = v_max
        if v_max <= 0:
            raise DataValidationError("v_max must be positive.")

    @property
    def columns(self) -> List[RouteColumn]:
        return list(self._columns)

    def solve_relaxation(self) -> tuple[float, Dict[int, float]]:
        model, lambdas, _ = self._build_model(binary=False)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError("RMP relaxation is infeasible or not optimal.")
        values = {idx: float(var.X) for idx, var in lambdas.items()}
        self._last_model = model
        self._last_cover_constraints = _
        return float(model.ObjVal), values

    def get_dual_variables(self) -> np.ndarray:
        """Return coverage-constraint duals \(\pi\) after LP optimization."""

        if not hasattr(self, "_last_cover_constraints"):
            raise RuntimeError("solve_relaxation() must be called before get_dual_variables().")
        return np.asarray([float(constr.Pi) for constr in self._last_cover_constraints], dtype=np.float64)

    def solve_integer(self) -> tuple[float | None, Dict[int, float] | None]:
        model, lambdas, _ = self._build_model(binary=True)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            return None, None
        return float(model.ObjVal), {idx: float(var.X) for idx, var in lambdas.items()}

    def _build_model(self, binary: bool) -> tuple[gp.Model, Dict[int, gp.Var], List[gp.Constr]]:
        model = gp.Model("q4_rmp")
        model.Params.OutputFlag = 0
        vtype = GRB.BINARY if binary else GRB.CONTINUOUS
        lambdas = {idx: model.addVar(lb=0.0, ub=1.0 if binary else GRB.INFINITY, vtype=vtype, name=f"lambda_{idx}") for idx in range(len(self._columns))}
        model.setObjective(gp.quicksum(self._columns[idx].cost * lambdas[idx] for idx in lambdas), GRB.MINIMIZE)
        cover_constraints: List[gp.Constr] = []
        for customer_id in self._instance.customer_ids:
            constr = model.addConstr(gp.quicksum(lambdas[idx] for idx, column in enumerate(self._columns) if customer_id in column.customers) >= 1.0, name=f"cover_{customer_id}")
            cover_constraints.append(constr)
        model.addConstr(gp.quicksum(lambdas.values()) <= self._v_max, name="vehicle_limit")
        return model, lambdas, cover_constraints


class PricingSubproblem:
    """Pricing engine with exact 0-1 knapsack reduction and local route search.

    Node filtering first solves a capacity-constrained knapsack DP with a hard
    cardinality cap of 15 and a preference for 10-15 selected nodes. The chosen
    subgraph is then routed by `AsyncClusterSolver`, and the reduced cost is
    computed against the current dual vector.
    """

    def __init__(self, instance: LogisticsInstance, compatibility_graph: TemporalCompatibilityGraph, duals: np.ndarray) -> None:
        self._instance = instance
        self._compatibility_graph = compatibility_graph
        self._duals = duals.astype(np.float64)
        self._capacity = int(round(instance.capacity))

    def solve(self) -> PricingResult:
        node_scores = self._node_scores()
        selected = self._solve_knapsack(node_scores)
        if not selected:
            return PricingResult([], None, 0.0)
        result = AsyncClusterSolver(self._instance, selected).solve(cluster_id=-1)
        column = self._build_column(result.route)
        reduced_cost = column.cost - float(np.sum(self._duals[np.asarray(column.customers) - 1]))
        return PricingResult(selected, column if reduced_cost < -1e-5 else None, reduced_cost)

    def _node_scores(self) -> Dict[int, float]:
        matrix = self._compatibility_graph.build_compatibility_matrix()
        positive = np.maximum(0.5 * (matrix + matrix.T), 0.0)
        degrees = np.sum(positive, axis=1)
        if np.max(degrees) > 0:
            degrees = degrees / np.max(degrees)
        return {node_id: float(self._duals[node_id - 1] + 0.2 * degrees[node_id - 1]) for node_id in self._instance.customer_ids}

    def _solve_knapsack(self, node_scores: Dict[int, float]) -> List[int]:
        """Exact 0-1 DP over capacity and selected-node count."""

        items = [(node.id, int(round(node.demand)), node_scores[node.id]) for node in self._instance.customers]
        max_count = min(15, len(items))
        dp = np.full((self._capacity + 1, max_count + 1), -np.inf, dtype=np.float64)
        choose: Dict[tuple[int, int, int], bool] = {}
        prev: Dict[tuple[int, int, int], tuple[int, int]] = {}
        dp[0, 0] = 0.0
        for item_idx, (node_id, weight, value) in enumerate(items, start=1):
            next_dp = dp.copy()
            for cap in range(weight, self._capacity + 1):
                for count in range(1, max_count + 1):
                    cand = dp[cap - weight, count - 1] + value
                    if cand > next_dp[cap, count]:
                        next_dp[cap, count] = cand
                        choose[(item_idx, cap, count)] = True
                        prev[(item_idx, cap, count)] = (cap - weight, count - 1)
            dp = next_dp
        admissible_counts = [c for c in range(10, max_count + 1)] or list(range(1, max_count + 1))
        best_cap, best_count, best_value = 0, 0, -np.inf
        for count in admissible_counts:
            cap = int(np.argmax(dp[:, count]))
            if dp[cap, count] > best_value:
                best_cap, best_count, best_value = cap, count, float(dp[cap, count])
        if best_count == 0 or not np.isfinite(best_value):
            return []
        selected: List[int] = []
        cap, count = best_cap, best_count
        for item_idx in range(len(items), 0, -1):
            key = (item_idx, cap, count)
            if key in choose:
                node_id, _, _ = items[item_idx - 1]
                selected.append(node_id)
                cap, count = prev[key]
                if count == 0:
                    break
        selected.reverse()
        return selected

    def _build_column(self, route: Sequence[int]) -> RouteColumn:
        if not route:
            raise DataValidationError("Pricing route cannot be empty.")
        load = float(sum(self._instance.get_node(node_id).demand for node_id in route))
        if load - self._instance.capacity > 1e-9:
            raise DataValidationError("Generated pricing route violates vehicle capacity.")
        first_travel = float(self._instance.travel_time_matrix[0, route[0]])
        current_time = first_travel
        travel_cost = first_travel
        penalties = self._node_penalty(route[0], current_time)
        for prev_id, node_id in zip(route[:-1], route[1:]):
            prev = self._instance.get_node(prev_id)
            travel = float(self._instance.travel_time_matrix[prev_id, node_id])
            travel_cost += travel
            current_time += prev.service_time + travel
            penalties += self._node_penalty(node_id, current_time)
        travel_cost += float(self._instance.travel_time_matrix[route[-1], 0])
        cost = travel_cost + penalties + VEHICLE_FIXED_COST
        return RouteColumn(route=list(route), customers=sorted(route), cost=cost, load=load)

    def _node_penalty(self, node_id: int, arrival_time: float) -> float:
        node = self._instance.get_node(node_id)
        early = max(node.ready_time - arrival_time, 0.0)
        late = max(arrival_time - node.due_time, 0.0)
        return 10.0 * early * early + 20.0 * late * late


class QCGEngine:
    """Quantum-classical column generation engine for problem 4.

    The engine alternates between solving the LP restricted master problem and
    generating negative reduced-cost routes through an exact knapsack reduction
    followed by the local asynchronous path optimizer inherited from problem 3.
    Integer recovery is performed only after root-node convergence.
    """

    def __init__(self, instance: LogisticsInstance, compatibility_graph: TemporalCompatibilityGraph, max_no_improve: int = 3, max_iterations: int = 30) -> None:
        self._instance = instance
        self._compatibility_graph = compatibility_graph
        self._max_no_improve = max_no_improve
        self._max_iterations = max_iterations
        self._columns: List[RouteColumn] = []

    def solve(self, v_max: int) -> EngineResult:
        self._columns = self._initialize_columns(v_max)
        no_improve = 0
        objective_lp = float("inf")
        lambda_lp: Dict[int, float] = {}
        for _ in range(self._max_iterations):
            master = MasterProblem(self._instance, self._columns, v_max)
            objective_lp, lambda_lp = master.solve_relaxation()
            duals = master.get_dual_variables()
            pricing = PricingSubproblem(self._instance, self._compatibility_graph, duals).solve()
            if pricing.column is None or self._column_exists(pricing.column):
                no_improve += 1
                if no_improve >= self._max_no_improve:
                    break
                continue
            self._columns.append(pricing.column)
            no_improve = 0
        objective_ip, lambda_ip = self.recover_integer_solution(v_max)
        return EngineResult(objective_lp, objective_ip, list(self._columns), lambda_lp, lambda_ip)

    def recover_integer_solution(self, v_max: int) -> tuple[float | None, Dict[int, float] | None]:
        """Heuristic Price-and-Branch recovery on the final column pool."""

        master = MasterProblem(self._instance, self._columns, v_max)
        return master.solve_integer()

    def run_sensitivity_analysis(self, vehicle_limits: Iterable[int]) -> Dict[int, float | None]:
        """Evaluate the final integer objective under multiple `V_max` values."""

        results: Dict[int, float | None] = {}
        for v_max in vehicle_limits:
            result = self.solve(v_max)
            results[int(v_max)] = result.objective_ip
        return results

    def _initialize_columns(self, v_max: int) -> List[RouteColumn]:
        """Build an initial feasible pool via sequential cuts plus artificial columns."""

        columns: List[RouteColumn] = []
        remaining = sorted(self._instance.customer_ids, key=lambda node_id: self._instance.get_node(node_id).ready_time)
        while remaining:
            route: List[int] = []
            load = 0.0
            for node_id in list(remaining):
                demand = self._instance.get_node(node_id).demand
                if load + demand <= self._instance.capacity:
                    route.append(node_id)
                    load += demand
                    remaining.remove(node_id)
            columns.append(self._build_initial_column(route))
        for node_id in self._instance.customer_ids:
            columns.append(RouteColumn(route=[node_id], customers=[node_id], cost=BIG_M_PENALTY, load=self._instance.get_node(node_id).demand))
        return columns

    def _build_initial_column(self, route: Sequence[int]) -> RouteColumn:
        result = AsyncClusterSolver(self._instance, route).solve(cluster_id=-1)
        travel_cost = float(self._depot_augmented_cost(result.route))
        cost = travel_cost + result.total_penalty + VEHICLE_FIXED_COST
        load = float(sum(self._instance.get_node(node_id).demand for node_id in result.route))
        return RouteColumn(route=list(result.route), customers=sorted(result.route), cost=cost, load=load)

    def _depot_augmented_cost(self, route: Sequence[int]) -> float:
        if not route:
            return 0.0
        cost = float(self._instance.travel_time_matrix[0, route[0]])
        for prev_id, node_id in zip(route[:-1], route[1:]):
            cost += float(self._instance.travel_time_matrix[prev_id, node_id])
        cost += float(self._instance.travel_time_matrix[route[-1], 0])
        return cost

    def _column_exists(self, candidate: RouteColumn) -> bool:
        signature = tuple(candidate.route)
        return any(tuple(column.route) == signature for column in self._columns)


__all__: List[str] = ["EngineResult", "MasterProblem", "PricingResult", "PricingSubproblem", "QCGEngine", "RouteColumn"]
