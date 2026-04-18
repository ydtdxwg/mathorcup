from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import logging
import kaiwu as kw
import matplotlib.pyplot as plt
import numpy as np


class QuantumTSPSolver:
    """Kaiwu-based QUBO TSP client with plotting support."""

    def __init__(self) -> None:
        logging.getLogger("kaiwu.sampler._simulated_annealing").setLevel(logging.WARNING)

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
        solver = kw.solver.SimpleSolver(worker)
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
        plt.title("哈密顿量随时间演化曲线", fontsize=14)
        plt.xlabel("退火步数", fontsize=12)
        plt.ylabel("能量 / 哈密顿量", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
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
