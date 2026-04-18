import json
from pathlib import Path


OUTPUT_DIR = Path("advanced_solver_outputs")
Q3_PATH = OUTPUT_DIR / "q3_results.json"
Q4_PATH = OUTPUT_DIR / "q4_results.json"
COMBINED_PATH = OUTPUT_DIR / "advanced_solver_results.json"


def _load_results() -> tuple[dict, dict]:
    if Q3_PATH.exists() and Q4_PATH.exists():
        with open(Q3_PATH, "r", encoding="utf-8") as f:
            q3_data = json.load(f)
        with open(Q4_PATH, "r", encoding="utf-8") as f:
            q4_data = json.load(f)
        return q3_data, q4_data

    with open(COMBINED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, data


def print_violations() -> None:
    q3_data, q4_data = _load_results()

    print("====== 问题 3 结果提取 ======")
    print(f"总运输时间: {q3_data['q3']['travel_cost']}")
    print(f"全局单车路线: {q3_data['q3']['route']}")
    print("\n--- Q3 各客户时间窗违反程度 (局部簇内核算) ---")
    for cluster in q3_data["q3"]["cluster_results"]:
        for node_id_str, arrival in cluster["arrival_times"].items():
            penalty = cluster["penalties"][node_id_str]
            if penalty > 0:
                print(f"客户 {node_id_str}: 产生二次惩罚 {penalty}")

    print("\n====== 问题 4 结果提取 (V_max = 5) ======")
    best_q4 = q4_data["q4"]["detailed_results"]["5"]
    active_lambda_ip = best_q4["active_lambda_ip"] or {}
    active_indices = [int(k) for k, v in active_lambda_ip.items() if v > 0.5]
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
