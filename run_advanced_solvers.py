from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data_pipeline import DataLoader, TemporalCompatibilityGraph
from src.q3_solver import Q3Solver
from src.q4_solver import QCGEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="运行问题三/问题四高级求解器")
    parser.add_argument("--task", choices=["q3", "q4", "both"], default="both")
    args = parser.parse_args()

    start_time = time.perf_counter()
    output_dir = Path("advanced_solver_outputs")
    output_dir.mkdir(exist_ok=True)
    all_logs: list[str] = []
    phase_logs: dict[str, list[str]] = {"q3": [], "q4": []}
    current_phase: str | None = None

    def log(message: str) -> None:
        nonlocal current_phase
        print(message)
        all_logs.append(message)
        if current_phase is None:
            phase_logs["q3"].append(message)
            phase_logs["q4"].append(message)
        else:
            phase_logs[current_phase].append(message)

    def save_log(path: Path, lines: list[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def q3_progress(event: dict) -> None:
        stage = event.get("stage")
        if stage == "cluster_partition_ready":
            log(f"[Q3 实时] 聚类完成 | 子簇数={event['cluster_count']} | 簇规模={event['cluster_sizes']}")
        elif stage == "cluster_start":
            log(f"[Q3 实时] 开始子簇 {event['cluster_index']}/{event['cluster_count']} | cluster_id={event['cluster_id']} | 节点数={event['node_count']}")
        elif stage == "cluster_iteration":
            if event["solver_mode"] == "quantum":
                log(f"[Q3 实时] 子簇 {event['cluster_id']} 迭代 {event['iteration']}/{event['max_iterations']} | mode=quantum | objective={event['objective']:.2f}")
            elif event["iteration"] == 1:
                log(f"[Q3 实时] 子簇 {event['cluster_id']} 已切换经典回退 | 原因={event['solver_message']}")
            elif event["iteration"] == event["max_iterations"] or event["iteration"] % 5 == 0:
                log(f"[Q3 实时] 子簇 {event['cluster_id']} 迭代 {event['iteration']}/{event['max_iterations']} | mode={event['solver_mode']} | objective={event['objective']:.2f}")
        elif stage == "cluster_complete":
            state = "Kaiwu成功" if event["quantum_used"] else "经典回退"
            log(f"[Q3 实时] 完成子簇 {event['cluster_index']}/{event['cluster_count']} | cluster_id={event['cluster_id']} | iterations={event['iterations']} | objective={event['objective']:.2f} | 量子状态={state}")
            log(f"            -> {event['quantum_message']}")
        elif stage == "global_stitch_start":
            log(f"[Q3 实时] 开始全局拼接 | 子簇数={event['cluster_count']}")
        elif stage == "global_stitch_complete":
            log(f"[Q3 实时] 全局拼接完成 | objective={event['objective']:.2f} | travel={event['travel_cost']:.2f} | penalty={event['total_penalty']:.2f} | route_length={event['route_length']}")

    log("=== 初始化数据管线 ===")
    excel_path = Path("参考算例.xlsx")
    instance = DataLoader(excel_path).load()
    total_demand = sum(node.demand for node in instance.customers)
    log(f"数据加载成功! 客户节点数: {instance.customer_count}, 车辆额定容量: {instance.capacity}")

    log("\n=== 构建时空相容性图 ===")
    alpha = 1.0
    beta = 1.0
    comp_graph = TemporalCompatibilityGraph(instance, alpha=alpha, beta=beta)
    log("时空相容性图构建完成")

    base_payload = {
        "excel_path": str(excel_path),
        "instance": {"customer_count": instance.customer_count, "capacity": instance.capacity, "total_demand": total_demand},
        "compatibility_parameters": {"alpha": alpha, "beta": beta},
    }

    q3_payload: dict[str, object] | None = None
    q4_payload: dict[str, object] | None = None

    if args.task in {"q3", "both"}:
        current_phase = "q3"
        log("\n=== 开始求解第三问 (50节点单车时空聚类-量子协同) ===")
        q3_start = time.perf_counter()
        q3_solution = Q3Solver(instance, comp_graph, n_clusters=5, local_max_iterations=15, progress_callback=q3_progress).solve()
        q3_elapsed = time.perf_counter() - q3_start
        q3_quantum_clusters = sum(1 for item in q3_solution.cluster_results if item.quantum_used)
        log(f"[Q3 进度] 聚类后子簇数: {len(q3_solution.cluster_results)}")
        log(f"[Q3 量子状态] 成功调用Kaiwu的子簇数: {q3_quantum_clusters}/{len(q3_solution.cluster_results)}")
        log(f"[Q3 完成] 用时: {q3_elapsed:.2f} 秒")
        log(f"[Q3 结果] 总目标值(成本+惩罚): {q3_solution.global_result.objective:.2f}")
        log(f"[Q3 结果] 纯运输时间: {q3_solution.global_result.travel_cost:.2f}")
        log(f"[Q3 结果] 时间窗惩罚: {q3_solution.global_result.total_penalty:.2f}")
        log(f"[Q3 结果] 全局路线: {q3_solution.global_result.route}")

        q3_payload = {
            **base_payload,
            "timing": {"q3_seconds": q3_elapsed},
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
                    }
                    for result in q3_solution.cluster_results
                ],
            },
        }
        (output_dir / "q3_results.json").write_text(json.dumps(q3_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        save_log(output_dir / "q3_run.log", phase_logs["q3"])
        log(f"[Q3 保存] 结果 JSON 已保存到: {output_dir / 'q3_results.json'}")
        log(f"[Q3 保存] 详细日志已保存到: {output_dir / 'q3_run.log'}")

    if args.task in {"q4", "both"}:
        current_phase = "q4"
        log("\n=== 开始求解第四问 (50节点多车受限 - 列生成) ===")
        q4_engine = QCGEngine(instance, comp_graph, max_iterations=20)
        min_vehicles = int((total_demand + instance.capacity - 1) // instance.capacity)
        test_vehicle_limits = [min_vehicles + 3, min_vehicles + 2, min_vehicles + 1, min_vehicles]
        log(f"系统总需求: {total_demand}, 理论最小车辆数: {min_vehicles}")
        log(f"开始扫描车辆数约束帕累托前沿: {test_vehicle_limits}")
        log("[Q4 提示] 列生成阶段是阻塞求解，单个 V_max 完成前可能暂时没有新输出。")

        sensitivity_results: dict[int, float | None] = {}
        q4_detailed_results: dict[int, dict[str, object]] = {}
        q4_seconds: dict[int, float] = {}
        for idx, v_max in enumerate(test_vehicle_limits, start=1):
            log(f"[Q4 进度] 正在求解 {idx}/{len(test_vehicle_limits)} : V_max = {v_max}")
            log(f"[Q4 提示] 已进入 V_max = {v_max} 的阻塞求解阶段。")
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
                "columns": [{"route": col.route, "customers": col.customers, "cost": col.cost, "load": col.load} for col in result.columns],
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
            if result.objective_ip is not None:
                log(f"[Q4 完成] V_max = {v_max} | 用时: {case_elapsed:.2f} 秒 | LP目标值={result.objective_lp:.2f} | 整数目标值={result.objective_ip:.2f} | 列数={len(result.columns)}")
            else:
                log(f"[Q4 完成] V_max = {v_max} | 用时: {case_elapsed:.2f} 秒 | LP目标值={result.objective_lp:.2f} | 整数恢复失败 | 列数={len(result.columns)}")

        log("\n[Q4 敏感性分析结果]")
        for v_max, obj in sensitivity_results.items():
            if obj is not None:
                log(f" -> 允许最多 {v_max} 辆车时, 综合最优成本为: {obj:.2f}")
            else:
                log(f" -> 允许最多 {v_max} 辆车时, 无法找到可行解 (Infeasible)")

        q4_payload = {
            **base_payload,
            "timing": {"q4_seconds": {str(k): v for k, v in q4_seconds.items()}},
            "q4": {"tested_vehicle_limits": test_vehicle_limits, "sensitivity_results": sensitivity_results, "detailed_results": q4_detailed_results},
        }
        (output_dir / "q4_results.json").write_text(json.dumps(q4_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        save_log(output_dir / "q4_run.log", phase_logs["q4"])
        log(f"[Q4 保存] 结果 JSON 已保存到: {output_dir / 'q4_results.json'}")
        log(f"[Q4 保存] 详细日志已保存到: {output_dir / 'q4_run.log'}")

    current_phase = None
    total_elapsed = time.perf_counter() - start_time
    combined_payload = dict(base_payload)
    combined_payload["timing"] = {"total_seconds": total_elapsed}
    if q3_payload is not None:
        combined_payload["timing"]["q3_seconds"] = q3_payload["timing"]["q3_seconds"]
        combined_payload["q3"] = q3_payload["q3"]
    if q4_payload is not None:
        combined_payload["timing"]["q4_seconds"] = q4_payload["timing"]["q4_seconds"]
        combined_payload["q4"] = q4_payload["q4"]

    (output_dir / "advanced_solver_results.json").write_text(json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_log(output_dir / "advanced_solver_run.log", all_logs)
    log(f"\n总耗时: {total_elapsed:.2f} 秒")
    log(f"汇总 JSON 已保存到: {output_dir / 'advanced_solver_results.json'}")
    log(f"汇总日志已保存到: {output_dir / 'advanced_solver_run.log'}")


if __name__ == "__main__":
    main()
