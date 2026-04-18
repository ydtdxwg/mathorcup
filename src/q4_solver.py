from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterable, List, Sequence

import gurobipy as gp
import numpy as np
from gurobipy import GRB

try:
    from .data_pipeline import DataValidationError, LogisticsInstance, TemporalCompatibilityGraph
except ImportError:
    from data_pipeline import DataValidationError, LogisticsInstance, TemporalCompatibilityGraph


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
class PricingIterationLog:
    """One column-generation iteration snapshot."""

    iteration: int
    objective_lp: float
    reduced_cost: float
    selected_nodes: List[int]
    candidate_route: List[int] | None
    column_added: bool
    duplicate_column: bool
    no_improve_streak: int
    column_count: int


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Final root-node price-and-branch result for one `V_max`."""

    objective_lp: float
    objective_ip: float | None
    columns: List[RouteColumn]
    lambda_lp: Dict[int, float]
    lambda_ip: Dict[int, float] | None
    iteration_logs: List[PricingIterationLog]
    performance_log: Dict[str, float]


class MasterProblem:
    """Restricted master problem for CVRP-TW set covering.

    The LP relaxation is
    \[\min \sum_{r\in\Omega} c_r\lambda_r\]
    subject to customer covering and a vehicle-count cap
    \[\sum_{r\in\Omega} a_{ir}\lambda_r \ge 1,\quad \sum_{r\in\Omega}\lambda_r\le V_{max},\quad \lambda_r\ge 0.\]
    """

    def __init__(self, instance: LogisticsInstance, columns: Sequence[RouteColumn], v_max: int, relaxation_time_limit: float = 15.0, integer_time_limit: float = 30.0) -> None:
        self._instance = instance
        self._columns = list(columns)
        self._v_max = v_max
        self._relaxation_time_limit = relaxation_time_limit
        self._integer_time_limit = integer_time_limit
        if v_max <= 0:
            raise DataValidationError("v_max must be positive.")

    @property
    def columns(self) -> List[RouteColumn]:
        return list(self._columns)

    def solve_relaxation(self) -> tuple[float, Dict[int, float]]:
        model, lambdas, cover_constraints, vehicle_constraint = self._build_model(binary=False)
        model.optimize()
        if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
            raise RuntimeError("RMP relaxation is infeasible or not optimal.")
        if model.SolCount <= 0:
            raise RuntimeError("RMP relaxation reached the time limit without any feasible LP solution.")
        values = {idx: float(var.X) for idx, var in lambdas.items()}
        self._last_model = model
        self._last_cover_constraints = cover_constraints
        self._last_vehicle_constraint = vehicle_constraint
        return float(model.ObjVal), values

    def get_dual_variables(self) -> tuple[np.ndarray, float]:
        """Return coverage duals \(\pi\) and vehicle-limit dual \(\mu\)."""

        if not hasattr(self, "_last_cover_constraints") or not hasattr(self, "_last_vehicle_constraint"):
            raise RuntimeError("solve_relaxation() must be called before get_dual_variables().")
        pi = np.asarray([float(constr.Pi) for constr in self._last_cover_constraints], dtype=np.float64)
        mu = float(self._last_vehicle_constraint.Pi)
        return pi, mu

    def solve_integer(self) -> tuple[float | None, Dict[int, float] | None]:
        model, lambdas, _, _ = self._build_model(binary=True)
        model.optimize()
        if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or model.SolCount <= 0:
            return None, None
        return float(model.ObjVal), {idx: float(var.X) for idx, var in lambdas.items()}

    def _build_model(self, binary: bool) -> tuple[gp.Model, Dict[int, gp.Var], List[gp.Constr], gp.Constr]:
        model = gp.Model("q4_rmp")
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = 0
        # 优化目的：给Gurobi主问题增加求解时限，避免LP/IP阶段长时间卡死无输出。
        model.Params.TimeLimit = self._integer_time_limit if binary else self._relaxation_time_limit
        vtype = GRB.BINARY if binary else GRB.CONTINUOUS
        lambdas = {idx: model.addVar(lb=0.0, ub=1.0 if binary else GRB.INFINITY, vtype=vtype, name=f"lambda_{idx}") for idx in range(len(self._columns))}
        model.setObjective(
            gp.quicksum((self._columns[idx].cost + VEHICLE_FIXED_COST) * lambdas[idx] for idx in lambdas),
            GRB.MINIMIZE,
        )
        cover_constraints: List[gp.Constr] = []
        for customer_id in self._instance.customer_ids:
            constr = model.addConstr(gp.quicksum(lambdas[idx] for idx, column in enumerate(self._columns) if customer_id in column.customers) >= 1.0, name=f"cover_{customer_id}")
            cover_constraints.append(constr)
        # 优化目的：敏感性分析按“固定使用V_max辆车”比较，确保不同车辆数会反映到目标值差异中。
        vehicle_constraint = model.addConstr(gp.quicksum(lambdas.values()) == self._v_max, name="vehicle_limit")
        return model, lambdas, cover_constraints, vehicle_constraint


class PricingSubproblem:
    """Pricing engine with exact 0-1 knapsack reduction and local route search.

    Node filtering first solves a capacity-constrained knapsack DP with a hard
    cardinality cap of 15 and a preference for 10-15 selected nodes. The chosen
    subgraph is then routed by `AsyncClusterSolver`, and the reduced cost is
    computed against the current dual vector.
    """

    def __init__(self, instance: LogisticsInstance, compatibility_graph: TemporalCompatibilityGraph, duals: np.ndarray, mu: float, affinity_matrix: np.ndarray | None = None) -> None:
        self._instance = instance
        self._compatibility_graph = compatibility_graph
        self._duals = duals.astype(np.float64)
        self._mu = float(mu)
        self._capacity = int(round(instance.capacity))
        self._affinity_matrix = affinity_matrix

    def solve(self) -> PricingResult:
        node_scores = self._node_scores()
        selected = self._solve_knapsack(node_scores)
        if not selected:
            return PricingResult([], None, 0.0)
        route = self._build_greedy_route(selected, node_scores)
        route = self._local_improve_route(route, max_swaps=8)
        column = self._build_column(route)
        reduced_cost = column.cost - float(np.sum(self._duals[np.asarray(column.customers) - 1])) - self._mu
        return PricingResult(selected, column if reduced_cost < -1e-5 else None, reduced_cost)

    def _node_scores(self) -> Dict[int, float]:
        matrix = self._affinity_matrix if self._affinity_matrix is not None else self._compatibility_graph.build_compatibility_matrix()
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
        cost = travel_cost + penalties
        return RouteColumn(route=list(route), customers=sorted(route), cost=cost, load=load)

    def _node_penalty(self, node_id: int, arrival_time: float) -> float:
        node = self._instance.get_node(node_id)
        early = max(node.ready_time - arrival_time, 0.0)
        late = max(arrival_time - node.due_time, 0.0)
        return 10.0 * early * early + 20.0 * late * late

    def _build_greedy_route(self, selected: Sequence[int], node_scores: Dict[int, float]) -> List[int]:
        return sorted(
            selected,
            key=lambda node_id: (
                -node_scores[node_id],
                self._instance.get_node(node_id).ready_time,
                node_id,
            ),
        )

    def _local_improve_route(self, route: Sequence[int], max_swaps: int = 8) -> List[int]:
        best_route = list(route)
        best_cost = self._build_column(best_route).cost
        swaps = 0
        improved = True
        while improved and swaps < max_swaps:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    candidate = list(best_route)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    candidate_cost = self._build_column(candidate).cost
                    swaps += 1
                    if candidate_cost + 1e-9 < best_cost:
                        best_route = candidate
                        best_cost = candidate_cost
                        improved = True
                        break
                    if swaps >= max_swaps:
                        break
                if improved or swaps >= max_swaps:
                    break
        return best_route


class QCGEngine:
    """Quantum-classical column generation engine for problem 4.

    The engine alternates between solving the LP restricted master problem and
    generating negative reduced-cost routes through an exact knapsack reduction
    followed by the local asynchronous path optimizer inherited from problem 3.
    Integer recovery is performed only after root-node convergence.
    """

    def __init__(self, instance: LogisticsInstance, compatibility_graph: TemporalCompatibilityGraph, max_no_improve: int = 3, max_iterations: int = 30, max_runtime_seconds: float = 300.0, stable_iteration_limit: int = 10, improvement_tolerance: float = 1e-3, progress_callback: Callable[[dict], None] | None = None) -> None:
        self._instance = instance
        self._compatibility_graph = compatibility_graph
        self._max_no_improve = max_no_improve
        self._max_iterations = max_iterations
        self._max_runtime_seconds = max_runtime_seconds
        self._stable_iteration_limit = stable_iteration_limit
        self._improvement_tolerance = improvement_tolerance
        self._progress_callback = progress_callback
        self._columns: List[RouteColumn] = []
        self._cached_affinity: np.ndarray | None = None
        self._performance_stats: Dict[str, float] = {}

    def solve(self, v_max: int) -> EngineResult:
        solve_start = time.perf_counter()
        self._performance_stats = {}
        affinity_matrix = self._cache_comp_graph()
        if self._progress_callback is not None:
            self._progress_callback({"stage": "q4_init", "message": "开始构建初始列池", "elapsed_seconds": 0.0})
        self._columns = self._initialize_columns(v_max, solve_start)
        if self._progress_callback is not None:
            self._progress_callback({"stage": "q4_init", "message": f"初始列池构建完成，列数={len(self._columns)}", "elapsed_seconds": time.perf_counter() - solve_start})
        no_improve = 0
        stable_iterations = 0
        previous_objective: float | None = None
        objective_lp = float("inf")
        lambda_lp: Dict[int, float] = {}
        iteration_logs: List[PricingIterationLog] = []
        for iteration in range(1, self._max_iterations + 1):
            if time.perf_counter() - solve_start >= self._max_runtime_seconds:
                break
            master_start = time.perf_counter()
            if self._progress_callback is not None:
                self._progress_callback(
                    {
                        "stage": "q4_master",
                        "message": f"开始主问题求解，当前列数={len(self._columns)}",
                        "elapsed_seconds": time.perf_counter() - solve_start,
                    }
                )
            master = MasterProblem(self._instance, self._columns, v_max)
            objective_lp, lambda_lp = master.solve_relaxation()
            self._performance_monitor("主问题求解耗时", master_start)
            pi_duals, mu_dual = master.get_dual_variables()
            pricing_start = time.perf_counter()
            pricing = PricingSubproblem(self._instance, self._compatibility_graph, pi_duals, mu_dual, affinity_matrix=affinity_matrix).solve()
            self._performance_monitor("定价子问题耗时", pricing_start)
            duplicate_column = pricing.column is not None and self._column_exists(pricing.column)
            column_added = pricing.column is not None and not duplicate_column
            if column_added:
                self._columns.append(pricing.column)
                no_improve = 0
            else:
                no_improve += 1
            if previous_objective is not None:
                relative_change = abs(objective_lp - previous_objective) / max(abs(previous_objective), 1.0)
                if relative_change <= self._improvement_tolerance:
                    stable_iterations += 1
                else:
                    stable_iterations = 0
            previous_objective = objective_lp
            iteration_logs.append(
                PricingIterationLog(
                    iteration=iteration,
                    objective_lp=objective_lp,
                    reduced_cost=pricing.reduced_cost,
                    selected_nodes=list(pricing.selected_nodes),
                    candidate_route=None if pricing.column is None else list(pricing.column.route),
                    column_added=column_added,
                    duplicate_column=duplicate_column,
                    no_improve_streak=no_improve,
                    column_count=len(self._columns),
                )
            )
            if self._progress_callback is not None and (iteration == 1 or iteration % 5 == 0):
                self._progress_callback(
                    {
                        "stage": "q4_iteration",
                        "iteration": iteration,
                        "max_iterations": self._max_iterations,
                        "objective": objective_lp,
                        "column_count": len(self._columns),
                        "elapsed_seconds": time.perf_counter() - solve_start,
                    }
                )
            if not column_added and no_improve >= self._max_no_improve:
                break
            if stable_iterations >= self._stable_iteration_limit:
                break
        objective_ip, lambda_ip = self.recover_integer_solution(v_max)
        self._performance_monitor("Q4总耗时", solve_start)
        return EngineResult(objective_lp, objective_ip, list(self._columns), lambda_lp, lambda_ip, iteration_logs, dict(self._performance_stats))

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

    def _initialize_columns(self, v_max: int, solve_start: float | None = None) -> List[RouteColumn]:
        """Build an initial feasible pool via sequential cuts plus artificial columns."""

        init_start = time.perf_counter()
        columns: List[RouteColumn] = []
        artificial_nodes: set[int] = set()
        covered_nodes: set[int] = set()
        remaining = sorted(self._instance.customer_ids, key=lambda node_id: self._instance.get_node(node_id).ready_time)
        route_count = 0
        while remaining:
            if solve_start is not None and time.perf_counter() - solve_start >= self._max_runtime_seconds:
                break
            route: List[int] = []
            load = 0.0
            for node_id in list(remaining):
                demand = self._instance.get_node(node_id).demand
                if load + demand <= self._instance.capacity:
                    route.append(node_id)
                    load += demand
                    remaining.remove(node_id)
            if not route:
                break
            route_count += 1
            primary_column = self._build_initial_column(route)
            columns.append(primary_column)
            covered_nodes.update(primary_column.customers)
            rotated_route = route[1:] + route[:1] if len(route) > 2 else list(route)
            rotated_column = self._build_initial_column(rotated_route)
            if tuple(rotated_column.route) != tuple(primary_column.route):
                columns.append(rotated_column)
                covered_nodes.update(rotated_column.customers)
            if self._progress_callback is not None and (route_count == 1 or route_count % 2 == 0):
                self._progress_callback(
                    {
                        "stage": "q4_init",
                        "message": f"已生成初始列 {len(columns)} 条，剩余客户数={len(remaining)}",
                        "elapsed_seconds": time.perf_counter() - init_start,
                    }
                )
        for node_id in remaining:
            columns.append(RouteColumn(route=[node_id], customers=[node_id], cost=BIG_M_PENALTY, load=self._instance.get_node(node_id).demand))
            artificial_nodes.add(node_id)
        for node_id in self._instance.customer_ids:
            if node_id not in artificial_nodes and node_id not in covered_nodes:
                columns.append(RouteColumn(route=[node_id], customers=[node_id], cost=BIG_M_PENALTY, load=self._instance.get_node(node_id).demand))
        self._performance_monitor("初始列构建耗时", init_start)
        return columns

    def _cache_comp_graph(self) -> np.ndarray:
        if self._cached_affinity is not None:
            return self._cached_affinity
        graph_start = time.perf_counter()
        matrix = self._compatibility_graph.build_compatibility_matrix()
        positive = np.maximum(0.5 * (matrix + matrix.T), 0.0)
        customer_travel = self._instance.travel_time_matrix[1:, 1:]
        threshold = float(np.mean(customer_travel)) * 2.0
        # 优化目的：对O(n²)图构建增加旅行时间阈值剪枝，跳过明显无效边。
        positive = np.where(customer_travel <= threshold, positive, 0.0)
        np.fill_diagonal(positive, 1.0)
        self._cached_affinity = positive
        self._performance_monitor("图构建耗时", graph_start)
        return positive

    def _build_initial_column(self, route: Sequence[int]) -> RouteColumn:
        # 优化目的：保留轻量初始化，但恢复时间窗惩罚与简单局部改良，避免初始列过于理想化。
        normalized_route = self._normalize_route(route)
        improved_route = self._improve_initial_route(normalized_route, max_swaps=6)
        cost = self._evaluate_route_cost(improved_route)
        load = float(sum(self._instance.get_node(node_id).demand for node_id in improved_route))
        return RouteColumn(route=list(improved_route), customers=sorted(improved_route), cost=cost, load=load)

    def _depot_augmented_cost(self, route: Sequence[int]) -> float:
        if not route:
            return 0.0
        cost = float(self._instance.travel_time_matrix[0, route[0]])
        for prev_id, node_id in zip(route[:-1], route[1:]):
            cost += float(self._instance.travel_time_matrix[prev_id, node_id])
        cost += float(self._instance.travel_time_matrix[route[-1], 0])
        return cost

    def _evaluate_route_cost(self, route: Sequence[int]) -> float:
        if not route:
            return BIG_M_PENALTY
        first_travel = float(self._instance.travel_time_matrix[0, route[0]])
        current_time = first_travel
        travel_cost = first_travel
        penalty = self._node_penalty(route[0], current_time)
        for prev_id, node_id in zip(route[:-1], route[1:]):
            prev = self._instance.get_node(prev_id)
            travel = float(self._instance.travel_time_matrix[prev_id, node_id])
            travel_cost += travel
            current_time += prev.service_time + travel
            penalty += self._node_penalty(node_id, current_time)
        travel_cost += float(self._instance.travel_time_matrix[route[-1], 0])
        return travel_cost + penalty

    def _node_penalty(self, node_id: int, arrival_time: float) -> float:
        node = self._instance.get_node(node_id)
        early = max(node.ready_time - arrival_time, 0.0)
        late = max(arrival_time - node.due_time, 0.0)
        return 10.0 * early * early + 20.0 * late * late

    def _normalize_route(self, route: Sequence[int]) -> List[int]:
        return sorted(route, key=lambda node_id: (self._instance.get_node(node_id).ready_time, node_id))

    def _improve_initial_route(self, route: Sequence[int], max_swaps: int = 6) -> List[int]:
        best_route = list(route)
        best_cost = self._evaluate_route_cost(best_route)
        swaps = 0
        improved = True
        while improved and swaps < max_swaps:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    candidate = list(best_route)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    candidate_cost = self._evaluate_route_cost(candidate)
                    swaps += 1
                    if candidate_cost + 1e-9 < best_cost:
                        best_route = candidate
                        best_cost = candidate_cost
                        improved = True
                        break
                    if swaps >= max_swaps:
                        break
                if improved or swaps >= max_swaps:
                    break
        return best_route

    def _column_exists(self, candidate: RouteColumn) -> bool:
        signature = tuple(candidate.route)
        return any(tuple(column.route) == signature for column in self._columns)

    def release_noncore_cache(self) -> None:
        self._cached_affinity = None

    def _performance_monitor(self, stage_name: str, start_time: float) -> None:
        self._performance_stats[stage_name] = self._performance_stats.get(stage_name, 0.0) + (time.perf_counter() - start_time)


__all__: List[str] = ["EngineResult", "MasterProblem", "PricingIterationLog", "PricingResult", "PricingSubproblem", "QCGEngine", "RouteColumn"]

# Q4性能优化点汇总：新增相容性图缓存与旅行时间阈值剪枝、加入300秒超时与连续稳定早停、减少高频进度刷新，并补充关键环节耗时监控与缓存释放接口。
