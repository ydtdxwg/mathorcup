import json


def print_violations():
    with open("advanced_solver_outputs/advanced_solver_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("====== 问题 3 结果提取 ======")
    print(f"总运输时间: {data['q3']['travel_cost']}")
    print(f"全局单车路线: {data['q3']['route']}")
    print("\n--- Q3 各客户时间窗违反程度 (局部簇内核算) ---")
    for cluster in data["q3"]["cluster_results"]:
        for node_id_str, arrival in cluster["arrival_times"].items():
            penalty = cluster["penalties"][node_id_str]
            if penalty > 0:
                print(f"客户 {node_id_str}: 产生二次惩罚 {penalty}")

    print("\n====== 问题 4 结果提取 (V_max = 5) ======")
    best_q4 = data["q4"]["detailed_results"]["5"]
    active_indices = [int(k) for k, v in best_q4["active_lambda_ip"].items() if v > 0.5]
    print(f"最终使用车辆数: {len(active_indices)}")

    for i, col_idx in enumerate(active_indices):
        col = best_q4["columns"][col_idx]
        print(f"\n[车辆 {i + 1}]")
        print(f"  行驶路线: {col['route']}")
        print(f"  服务客户: {col['customers']}")
        print(f"  车辆载重: {col['load']} / 60")
        print(f"  单车综合成本(含固定费与惩罚): {col['cost']}")


if __name__ == "__main__":
    print_violations()
