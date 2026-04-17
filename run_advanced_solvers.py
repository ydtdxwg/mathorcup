from __future__ import annotations

import json
from pathlib import Path

from src.data_pipeline import DataLoader, TemporalCompatibilityGraph
from src.q3_solver import Q3Solver
from src.q4_solver import QCGEngine


def main() -> None:
    print("=== 初始化数据管线 ===")
    excel_path = Path("参考算例.xlsx")
    loader = DataLoader(excel_path)
    instance = loader.load()
    print(f"数据加载成功! 客户节点数: {instance.customer_count}, 车辆额定容量: {instance.capacity}")

    print("\n=== 构建时空相容性图 ===")
    alpha = 1.0
    beta = 1.0
    comp_graph = TemporalCompatibilityGraph(instance, alpha=alpha, beta=beta)

    print("\n=== 开始求解第三问 (50节点单车时空聚类-量子协同) ===")
    q3_solver = Q3Solver(instance, comp_graph, n_clusters=5, local_max_iterations=15)
    q3_solution = q3_solver.solve()

    print(f"[Q3 结果] 总目标值(成本+惩罚): {q3_solution.global_result.objective:.2f}")
    print(f"[Q3 结果] 纯运输时间: {q3_solution.global_result.travel_cost:.2f}")
    print(f"[Q3 结果] 时间窗惩罚: {q3_solution.global_result.total_penalty:.2f}")
    print(f"[Q3 结果] 全局路线: {q3_solution.global_result.route}")

    print("\n=== 开始求解第四问 (50节点多车受限 - 列生成) ===")
    q4_engine = QCGEngine(instance, comp_graph, max_iterations=20)

    total_demand = sum(node.demand for node in instance.customers)
    min_vehicles = int((total_demand + instance.capacity - 1) // instance.capacity)
    print(f"系统总需求: {total_demand}, 理论最小车辆数: {min_vehicles}")

    test_vehicle_limits = [min_vehicles + 3, min_vehicles + 2, min_vehicles + 1, min_vehicles]
    print(f"开始扫描车辆数约束帕累托前沿: {test_vehicle_limits}")
    sensitivity_results = q4_engine.run_sensitivity_analysis(test_vehicle_limits)

    print("\n[Q4 敏感性分析结果]")
    for v_max, obj in sensitivity_results.items():
        if obj is not None:
            print(f" -> 允许最多 {v_max} 辆车时, 综合最优成本为: {obj:.2f}")
        else:
            print(f" -> 允许最多 {v_max} 辆车时, 无法找到可行解 (Infeasible)")

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
        "q3": {
            "objective": q3_solution.global_result.objective,
            "travel_cost": q3_solution.global_result.travel_cost,
            "total_penalty": q3_solution.global_result.total_penalty,
            "route": q3_solution.global_result.route,
            "supernode_order": q3_solution.global_result.supernode_order,
        },
        "q4": {
            "tested_vehicle_limits": test_vehicle_limits,
            "sensitivity_results": sensitivity_results,
        },
    }

    output_path = Path("advanced_solver_results.json")
    output_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
