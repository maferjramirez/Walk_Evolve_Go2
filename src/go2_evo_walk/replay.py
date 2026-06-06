from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence
import math

import pybullet as p

from .sim import QuadrupedEvaluator, SimulationConfig


def load_genome(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"No encontré el archivo de entrada: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "genome" not in data:
        raise KeyError("Input JSON does not contain 'genome' field")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce un mejor genoma guardado en la GUI de PyBullet")
    parser.add_argument("--input", "-i", type=str, default="best_run.json", help="Ruta al JSON guardado con la corrida")
    parser.add_argument("--urdf", type=str, default=None, help="Ruta opcional para sobreescribir el URDF")
    parser.add_argument("--episode-seconds", type=float, default=10.0, help="Segundos a simular durante el replay")
    parser.add_argument("--settle-seconds", type=float, default=1.0, help="Segundos de asentamiento antes de reproducir")
    parser.add_argument("--slowdown", type=float, default=1.0, help="Factor para hacer más lenta la visualización (>1 más lento)")
    parser.add_argument("--hold", action="store_true", help="Dejo la GUI abierta después del replay hasta que presionen una tecla")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_genome(args.input)
    genome = data["genome"]
    urdf = args.urdf if args.urdf is not None else data.get("urdf")

    config = SimulationConfig(urdf_path=urdf, gui=True, episode_seconds=args.episode_seconds, settle_seconds=args.settle_seconds)
    evaluator = QuadrupedEvaluator(config)
    try:
        print(f"Reproduciendo el genoma desde: {args.input}")

        # Replay manual para poder bajar la velocidad y dejar la GUI abierta.
        evaluator._reset_world()
        robot_id = evaluator._load_robot()
        frequency, amplitudes, phases, offsets = evaluator._decode_genome(genome)

        start_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=evaluator.client_id)
        start_x = start_pos[0]
        start_y = start_pos[1]

        settle_steps = int(evaluator.config.settle_seconds / evaluator.config.timestep)
        motion_steps = int(args.episode_seconds / evaluator.config.timestep)
        executed_steps = 0
        sleep_time = max(0.0, evaluator.config.timestep * args.slowdown)

        for step in range(settle_steps + motion_steps):
            if step >= settle_steps:
                t = (step - settle_steps) * evaluator.config.timestep
                for gene_index, joint_index in enumerate(evaluator._joint_indices):
                    target = offsets[gene_index] + amplitudes[gene_index] * (
                        math.sin(2.0 * math.pi * frequency * t + phases[gene_index])
                    )
                    target = max(-1.4, min(1.4, target))
                    p.setJointMotorControl2(
                        robot_id,
                        joint_index,
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=evaluator.config.control_force,
                        positionGain=0.28,
                        velocityGain=1.0,
                        physicsClientId=evaluator.client_id,
                    )
                executed_steps += 1

            p.stepSimulation(physicsClientId=evaluator.client_id)
            time.sleep(sleep_time)

            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=evaluator.client_id)
            if base_pos[2] < evaluator.config.fall_height:
                break

        final_pos, final_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=evaluator.client_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(final_orn)
        forward_progress = final_pos[0] - start_x
        lateral_drift = abs(final_pos[1] - start_y)
        posture_penalty = abs(roll) + abs(pitch)
        fitness = forward_progress - 0.35 * lateral_drift - 0.5 * posture_penalty

        print(f"Replay terminado. Fitness aproximado: {fitness}")

        if args.hold:
            input("Presiona Enter para cerrar la GUI y salir...")

        return 0
    finally:
        evaluator.close()


if __name__ == "__main__":
    raise SystemExit(main())
