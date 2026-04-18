from __future__ import annotations

import json
import time
from pathlib import Path

from src.data_pipeline import DataLoader, TemporalCompatibilityGraph
from src.q3_solver import Q3Solver
from src.q4_solver import QCGEngine


def main() -> None:
    start_time = time.perf_counter()
    output_dir = Path("advanced_solver_outputs")
    output_dir.mkdir(exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    def q3_progress(event: dict) -> None:
        stage = event.get("stage")
        if stage == "cluster_partition_ready":
            log(
                f"[Q3 实时] 聚类完成 | 子簇数={event['cluster_count']} | "
                f"簇规模={event['cluster_sizes']}"
            )
        elif stage == "cluster_start":
            log(
                f"[Q3 实时] 开始子簇 {event['cluster_index']}/{event['cluster_count']} | "
                f"cluster_id={event['cluster_id']} | 节点数={event['node_count']} | 节点={event['node_ids']}"
            )
        elif stage == "cluster_iteration":
            log(
                f"[Q3 实时] 子簇 {event['cluster_id']} 迭代 {event['iteration']}/{event['max_iterations']} | "
                f"mode={event['solver_mode']} | objective={event['objective']:.2f} | "
                f"travel={event['travel_cost']:.2f} | penalty={event['total_penalty']:.2f}"
            )
            log(f"            -> {event['solver_message']}")
        elif stage == "cluster_complete":
            log(
                f"[Q3 实时] 完成子簇 {event['cluster_index']}/{event['cluster_count']} | "
                f"cluster_id={event['cluster_id']} | iterations={event['iterations']} | "
                f"objective={event['objective']:.2f} | "
                f"量子状态={'Kaiwu成功' if event['quantum_used'] else '经典回退'}"
            )
            log(f"            -> {event['quantum_message']}")
        elif stage == "global_stitch_start":
            log(f"[Q3 实时] 开始全局拼接 | 子簇数={event['cluster_count']}")
        elif stage == "global_stitch_complete":
            log(
                f"[Q3 实时] 全局拼接完成 | objective={event['objective']:.2f} | "
                f"travel={event['travel_cost']:.2f} | penalty={event['total_penalty']:.2f} | "
                f"route_length={event['route_length']}"
            )

    log("=== 初始化数据管线 ===")
    excel_path = Path("参考算例.xlsx")
    loader = DataLoader(excel_path)
    instance = loader.load()
    log(f"数据加载成功! 客户节点数: {instance.customer_count}, 车辆额定容量: {instance.capacity}")

    log("\n=== 构建时空相容性图 ===")
    alpha = 1.0
    beta = 1.0
    comp_graph = TemporalCompatibilityGraph(instance, alpha=alpha, beta=beta)
    log("时空相容性图构建完成")

    log("\n=== 开始求解第三问 (50节点单车时空聚类-量子协同) ===")
    q3_start = time.perf_counter()
    q3_solver = Q3Solver(instance, comp_graph, n_clusters=5, local_max_iterations=15, progress_callback=q3_progress)
    q3_solution = q3_solver.solve()
    q3_elapsed = time.perf_counter() - q3_start

    log(f"[Q3 进度] 聚类后子簇数: {len(q3_solution.cluster_results)}")
    q3_quantum_clusters = sum(1 for result in q3_solution.cluster_results if result.quantum_used)
    log(f"[Q3 量子状态] 成功调用Kaiwu的子簇数: {q3_quantum_clusters}/{len(q3_solution.cluster_results)}")
    for idx, cluster_result in enumerate(q3_solution.cluster_results, start=1):
        log(
            f"  - 子簇 {idx}/{len(q3_solution.cluster_results)} | "
            f"cluster_id={cluster_result.cluster_id} | "
            f"节点数={len(cluster_result.route)} | "
            f"迭代次数={cluster_result.iterations} | "
            f"局部目标值={cluster_result.objective:.2f} | "
            f"量子状态={'Kaiwu成功' if cluster_result.quantum_used else '经典回退'}"
        )
        log(f"      * 求解说明: {cluster_result.quantum_message}")
        for iteration_log in cluster_result.iteration_logs:
            log(
                f"      * Q3迭代 {iteration_log.iteration:02d} | "
                f"mode={iteration_log.solver_mode} | "
                f"travel={iteration_log.travel_cost:.2f} | "
                f"penalty={iteration_log.total_penalty:.2f} | "
                f"objective={iteration_log.objective:.2f} | "
                f"route={iteration_log.route}"
            )
            log(f"        -> {iteration_log.solver_message}")
    log(f"[Q3 完成] 用时: {q3_elapsed:.2f} 秒")
    log(f"[Q3 结果] 总目标值(成本+惩罚): {q3_solution.global_result.objective:.2f}")
    log(f"[Q3 结果] 纯运输时间: {q3_solution.global_result.travel_cost:.2f}")
    log(f"[Q3 结果] 时间窗惩罚: {q3_solution.global_result.total_penalty:.2f}")
    log(f"[Q3 结果] 全局路线: {q3_solution.global_result.route}")

    log("\n=== 开始求解第四问 (50节点多车受限 - 列生成) ===")
    q4_engine = QCGEngine(instance, comp_graph, max_iterations=20)

    total_demand = sum(node.demand for node in instance.customers)
    min_vehicles = int((total_demand + instance.capacity - 1) // instance.capacity)
    log(f"系统总需求: {total_demand}, 理论最小车辆数: {min_vehicles}")

    test_vehicle_limits = [min_vehicles + 3, min_vehicles + 2, min_vehicles + 1, min_vehicles]
    log(f"开始扫描车辆数约束帕累托前沿: {test_vehicle_limits}")
    sensitivity_results: dict[int, float | None] = {}
    q4_detailed_results: dict[int, dict[str, object]] = {}
    q4_seconds: dict[int, float] = {}
    for idx, v_max in enumerate(test_vehicle_limits, start=1):
        log(f"[Q4 进度] 正在求解 {idx}/{len(test_vehicle_limits)} : V_max = {v_max}")
        q4_case_start = time.perf_counter()
        result = q4_engine.solve(v_max)
        case_elapsed = time.perf_counter() - q4_case_start
        q4_seconds[v_max] = case_elapsed
        sensitivity_results[v_max] = result.objective_ip
        q4_detailed_results[v_max] = {
            "objective_lp": result.objective_lp,
            "objective_ip": result.objective_ip,
            "column_count": len(result.columns),
            "active_lambda_lp": {str(k): v for k, v in result.lambda_lp.items() if v > 1e-6},
            "active_lambda_ip": None if result.lambda_ip is None else {str(k): v for k, v in result.lambda_ip.items() if v > 1e-6},
            "columns": [
                {
                    "route": column.route,
                    "customers": column.customers,
                    "cost": column.cost,
                    "load": column.load,
                }
                for column in result.columns
            ],
            "iteration_logs": [
                {
                    "iteration": item.iteration,
                    "objective_lp": item.objective_lp,
                    "reduced_cost": item.reduced_cost,
                    "selected_nodes": item.selected_nodes,
                    "candidate_route": item.candidate_route,
                    "column_added": item.column_added,
                    "duplicate_column": item.duplicate_column,
                    "no_improve_streak": item.no_improve_streak,
                    "column_count": item.column_count,
                }
                for item in result.iteration_logs
            ],
        }
        for item in result.iteration_logs:
            log(
                f"      * Q4迭代 {item.iteration:02d} | "
                f"LP={item.objective_lp:.2f} | rc={item.reduced_cost:.2f} | "
                f"selected={item.selected_nodes} | route={item.candidate_route} | "
                f"added={item.column_added} | duplicate={item.duplicate_column} | "
                f"no_improve={item.no_improve_streak} | columns={item.column_count}"
            )
        if result.objective_ip is not None:
            log(
                f"[Q4 完成] V_max = {v_max} | 用时: {case_elapsed:.2f} 秒 | "
                f"LP目标值={result.objective_lp:.2f} | 整数目标值={result.objective_ip:.2f} | 列数={len(result.columns)}"
            )
        else:
            log(
                f"[Q4 完成] V_max = {v_max} | 用时: {case_elapsed:.2f} 秒 | "
                f"LP目标值={result.objective_lp:.2f} | 整数恢复失败 | 列数={len(result.columns)}"
            )

    log("\n[Q4 敏感性分析结果]")
    for v_max, obj in sensitivity_results.items():
        if obj is not None:
            log(f" -> 允许最多 {v_max} 辆车时, 综合最优成本为: {obj:.2f}")
        else:
            log(f" -> 允许最多 {v_max} 辆车时, 无法找到可行解 (Infeasible)")

    total_elapsed = time.perf_counter() - start_time
    result_payload = {
        "excel_path": str(excel_path),
        "instance": {
            "customer_count": instance.customer_count,
            "capacity": instance.capacity,
            "total_demand": total_demand,
        },
        "compatibility_parameters": {
            "alpha": alpha,
            "beta": beta,
        },
        "timing": {
            "q3_seconds": q3_elapsed,
            "q4_seconds": {str(k): v for k, v in q4_seconds.items()},
            "total_seconds": total_elapsed,
        },
        "q3": {
            "objective": q3_solution.global_result.objective,
            "travel_cost": q3_solution.global_result.travel_cost,
            "total_penalty": q3_solution.global_result.total_penalty,
            "route": q3_solution.global_result.route,
            "supernode_order": q3_solution.global_result.supernode_order,
            "cluster_results": [
                {
                    "cluster_id": result.cluster_id,
                    "route": result.route,
                    "iterations": result.iterations,
                    "travel_cost": result.travel_cost,
                    "total_penalty": result.total_penalty,
                    "objective": result.objective,
                    "quantum_used": result.quantum_used,
                    "quantum_message": result.quantum_message,
                    "arrival_times": {str(k): v for k, v in result.arrival_times.items()},
                    "penalties": {str(k): v for k, v in result.penalties.items()},
                    "iteration_logs": [
                        {
                            "iteration": item.iteration,
                            "route": item.route,
                            "travel_cost": item.travel_cost,
                            "total_penalty": item.total_penalty,
                            "objective": item.objective,
                            "solver_mode": item.solver_mode,
                            "solver_message": item.solver_message,
                        }
                        for item in result.iteration_logs
                    ],
                }
                for result in q3_solution.cluster_results
            ],
        },
        "q4": {
            "tested_vehicle_limits": test_vehicle_limits,
            "sensitivity_results": sensitivity_results,
            "detailed_results": q4_detailed_results,
        },
    }

    json_output_path = output_dir / "advanced_solver_results.json"
    json_output_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_output_path = output_dir / "advanced_solver_run.log"
    log_output_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"\n总耗时: {total_elapsed:.2f} 秒")
    log(f"结果 JSON 已保存到: {json_output_path}")
    log(f"详细日志已保存到: {log_output_path}")
    log_output_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
