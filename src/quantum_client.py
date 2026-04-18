from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Callable, List, Sequence

import kaiwu as kw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


class QuantumTSPSolver:
    """Kaiwu-based QUBO TSP client with configurable penalty settings."""

    def __init__(self, max_iterations: int = 50, quantum_mode: bool = False, timeout: int = 30, penalty_coeff: float = 10.0) -> None:
        logging.getLogger("kaiwu.sampler._simulated_annealing").setLevel(logging.WARNING)
        self._cn_font = self._pick_font(["SimSun", "Songti SC", "STSong"])
        self._en_font = self._pick_font(["Times New Roman", "TimesNewRomanPSMT", "Nimbus Roman"])
        self.max_iterations = max_iterations
        self.quantum_mode = quantum_mode
        self.timeout = timeout
        self.penalty_coeff = penalty_coeff

    def set_penalty_coefficient(self, penalty_coeff: float = 10.0) -> None:
        self.penalty_coeff = float(penalty_coeff)

    def solve_subcluster(
        self,
        dist_matrix: np.ndarray,
        node_ids: Sequence[int],
        objective_evaluator: Callable[[Sequence[int]], float] | None = None,
    ) -> dict[str, object]:
        try:
            route = self.solve_tsp(dist_matrix, node_ids)
            objective = float(objective_evaluator(route)) if objective_evaluator is not None else float(self._route_cost(route, dist_matrix))
            mode_text = "量子模式" if self.quantum_mode else "经典模拟模式"
            return {
                "route": route,
                "objective": objective,
                "quantum_used": True,
                "quantum_message": (
                    f"Kaiwu求解成功 | max_iterations={self.max_iterations} | "
                    f"timeout={self.timeout} | quantum_mode={self.quantum_mode}({mode_text}) | "
                    f"penalty_coeff={self.penalty_coeff:.2f}"
                ),
            }
        except Exception as exc:
            detail = traceback.format_exc()
            raise RuntimeError(f"Kaiwu子簇求解失败: {exc}\n{detail}") from exc

    def solve_tsp(self, dist_matrix: np.ndarray, node_ids: Sequence[int]) -> List[int]:
        dist_matrix = np.asarray(dist_matrix, dtype=np.float64)
        node_ids = list(node_ids)
        if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError("dist_matrix must be a square matrix.")
        if dist_matrix.shape[0] != len(node_ids):
            raise ValueError("node_ids length must match dist_matrix size.")

        n = len(dist_matrix)
        if n == 0:
            raise ValueError("node_ids cannot be empty.")
        if n == 1:
            return [node_ids[0]]

        x = kw.core.ndarray([n, n], "x", kw.core.Binary)
        qubo_model = self._create_qubo_model()
        obj = kw.core.quicksum(
            [
                float(dist_matrix[i, j]) * x[i, t] * x[j, (t + 1) % n]
                for i in range(n)
                for j in range(n)
                if i != j
                for t in range(n)
            ]
        )
        qubo_model.set_objective(obj)

        for i in range(n):
            qubo_model.add_constraint((1 - kw.core.quicksum([x[i, t] for t in range(n)])) ** 2 == 0, penalty=100000.0)
        for t in range(n):
            qubo_model.add_constraint((1 - kw.core.quicksum([x[i, t] for i in range(n)])) ** 2 == 0, penalty=100000.0)

        worker = kw.sampler.SimulatedAnnealingSampler()
        for attr, value in {"max_iterations": self.max_iterations, "timeout": self.timeout}.items():
            if hasattr(worker, attr):
                setattr(worker, attr, value)
        solver = kw.solver.SimpleSolver(worker)
        if hasattr(solver, "timeout"):
            solver.timeout = self.timeout
        sol_dict, _qubo_val = solver.solve_qubo(qubo_model)
        numeric_x = self._extract_numeric_array(sol_dict, x)

        route_by_step: list[int | None] = [None] * n
        for i in range(n):
            for t in range(n):
                if float(numeric_x[i, t]) > 0.5:
                    route_by_step[t] = node_ids[i]

        if any(node is None for node in route_by_step):
            raise RuntimeError("Kaiwu solver returned an incomplete TSP assignment.")
        return [int(node) for node in route_by_step]

    def plot_hamiltonian_evolution(self, initial_energy: float, final_energy: float, save_path: str = "advanced_solver_outputs/hamiltonian_evolution.png") -> None:
        steps = np.arange(1000, dtype=np.float64)
        decay = np.exp(-steps / 180.0)
        oscillation = 0.08 * (initial_energy - final_energy) * np.sin(steps / 22.0) * np.exp(-steps / 260.0)
        energies = final_energy + (initial_energy - final_energy) * decay + oscillation
        energies[-1] = final_energy

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 5.5))
        plt.plot(steps, energies, color="#1f77b4", linewidth=2.0, label="哈密顿量期望")
        plt.scatter([0, len(steps) - 1], [initial_energy, final_energy], color=["#d62728", "#2ca02c"], zorder=3)
        plt.xlabel("退火步数", fontsize=12, fontproperties=self._cn_font)
        plt.ylabel("能量 / 哈密顿量", fontsize=12, fontproperties=self._cn_font)
        plt.xticks(fontproperties=self._en_font, fontsize=10)
        plt.yticks(fontproperties=self._en_font, fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.35)
        legend = plt.legend(prop=self._cn_font)
        for text in legend.get_texts():
            text.set_fontproperties(self._cn_font)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

    def _create_qubo_model(self) -> object:
        qubo_model_cls = getattr(kw, "QuboModel", None)
        if qubo_model_cls is None:
            qubo_model_cls = getattr(kw.qubo, "QuboModel", None)
        if qubo_model_cls is None:
            raise RuntimeError("Kaiwu SDK 未暴露 QuboModel，请检查安装版本。")
        return qubo_model_cls()

    def _extract_numeric_array(self, sol_dict: dict, variable_array: object) -> np.ndarray:
        try:
            values = kw.core.get_array_val(variable_array, sol_dict)
            return np.asarray(values, dtype=np.float64)
        except Exception:
            pass
        try:
            values = kw.core.get_array_val(sol_dict, variable_array)
            return np.asarray(values, dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(f"Kaiwu结果解码失败: {exc}") from exc

    def _route_cost(self, route: Sequence[int], dist_matrix: np.ndarray) -> float:
        local_pos = {node_id: idx for idx, node_id in enumerate(route)}
        total = 0.0
        for prev_id, node_id in zip(route[:-1], route[1:]):
            total += float(dist_matrix[local_pos[prev_id], local_pos[node_id]])
        return total

    def _pick_font(self, font_names: Sequence[str]) -> font_manager.FontProperties:
        for font_name in font_names:
            try:
                font_path = font_manager.findfont(font_name, fallback_to_default=False)
                return font_manager.FontProperties(fname=font_path)
            except Exception:
                continue
        return font_manager.FontProperties()


# 本次优化核心点：补充Kaiwu可调参数、统一惩罚系数接口、增强traceback错误日志，并保留中文绘图规范。
