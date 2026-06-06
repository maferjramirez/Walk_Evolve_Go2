from __future__ import annotations

import math
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from contextlib import contextmanager

import numpy as np
import pybullet as p
import pybullet_data


@dataclass(frozen=True)
class SimulationConfig:
    urdf_path: str | None = None
    gui: bool = False
    episode_seconds: float = 6.0
    settle_seconds: float = 1.0
    timestep: float = 1.0 / 240.0
    control_force: float = 30.0
    fall_height: float = 0.16


class QuadrupedEvaluator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.client_id = p.connect(p.GUI if config.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(fixedTimeStep=config.timestep, numSolverIterations=150, physicsClientId=self.client_id)
        self._robot_urdf = self._resolve_robot_urdf(config.urdf_path)
        self._joint_indices: list[int] = []
        self._robot_id: int | None = None
        self._reset_world()
        self._load_robot()

    @contextmanager
    def _suppress_bullet_output(self):
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(devnull_fd)
            os.close(stdout_fd)
            os.close(stderr_fd)

    @property
    def gene_count(self) -> int:
        return 1 + 3 * len(self._joint_indices)

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def _resolve_robot_urdf(self, urdf_path: str | None) -> str:
        if urdf_path:
            path = Path(urdf_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"URDF not found: {path}")
            return str(path)

        candidate = Path(pybullet_data.getDataPath()) / "laikago" / "laikago_toes.urdf"
        if candidate.exists():
            return str(candidate)

        raise FileNotFoundError(
            "No default quadruped URDF found. Pass --urdf with the Unitree Go2 model path."
        )

    def _reset_world(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.changeDynamics(plane_id, -1, lateralFriction=1.0, physicsClientId=self.client_id)

    def _load_robot(self) -> int:
        start_position = [0.0, 0.0, 0.32]
        start_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        with self._suppress_bullet_output():
            robot_id = p.loadURDF(
                self._robot_urdf,
                start_position,
                start_orientation,
                flags=p.URDF_USE_SELF_COLLISION | p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.client_id,
            )

        joint_indices: list[int] = []
        for joint_index in range(p.getNumJoints(robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(robot_id, joint_index, physicsClientId=self.client_id)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                joint_name = info[1].decode("utf-8")
                if joint_name.endswith("_hip_joint"):
                    neutral_position = 0.0
                elif joint_name.endswith("_thigh_joint"):
                    neutral_position = 0.8
                elif joint_name.endswith("_calf_joint"):
                    neutral_position = -1.6
                else:
                    neutral_position = 0.0

                joint_indices.append(joint_index)
                p.resetJointState(robot_id, joint_index, targetValue=neutral_position, targetVelocity=0.0, physicsClientId=self.client_id)
                p.setJointMotorControl2(
                    robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=neutral_position,
                    force=self.config.control_force,
                    physicsClientId=self.client_id,
                )

        if not joint_indices:
            raise RuntimeError("The loaded URDF has no revolute joints to control.")

        self._joint_indices = joint_indices
        self._robot_id = robot_id
        return robot_id

    def _decode_genome(self, genome: Sequence[float]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        # Codificación del genoma: real.
        # Uso una frecuencia global y, por cada articulación, una amplitud, una fase y un offset.
        # Las multiplicaciones escalan cada variable a rangos útiles para la marcha:
        # - frecuencia: la llevo a un rango de movimiento razonable para el ciclo de la pisada.
        # - amplitudes: limito cuánto se abre cada articulación para evitar movimientos exagerados.
        # - fases: las paso a [-pi, pi] para cubrir un ciclo completo de desfasaje.
        # - offsets: los reduzco para mantener la postura cerca del centro y no forzar la articulación.
        joint_count = len(self._joint_indices)
        if len(genome) != self.gene_count:
            raise ValueError(f"Expected {self.gene_count} genes, got {len(genome)}")

        frequency = 0.5 + 1.5 * (math.tanh(genome[0]) + 1.0) * 0.5
        amplitudes = 0.8 * np.tanh(np.asarray(genome[1 : 1 + joint_count], dtype=float))
        phases = math.pi * np.tanh(np.asarray(genome[1 + joint_count : 1 + 2 * joint_count], dtype=float))
        offsets = 0.4 * np.tanh(np.asarray(genome[1 + 2 * joint_count : 1 + 3 * joint_count], dtype=float))
        return frequency, amplitudes, phases, offsets

    def evaluate(self, genome: Sequence[float]) -> tuple[float]:
        self._reset_world()
        robot_id = self._load_robot()
        frequency, amplitudes, phases, offsets = self._decode_genome(genome)

        start_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client_id)
        start_x = start_pos[0]
        start_y = start_pos[1]

        settle_steps = int(self.config.settle_seconds / self.config.timestep)
        motion_steps = int(self.config.episode_seconds / self.config.timestep)
        total_action = 0.0
        executed_steps = 0

        for step in range(settle_steps + motion_steps):
            if step >= settle_steps:
                t = (step - settle_steps) * self.config.timestep
                for gene_index, joint_index in enumerate(self._joint_indices):
                    target = offsets[gene_index] + amplitudes[gene_index] * math.sin(
                        2.0 * math.pi * frequency * t + phases[gene_index]
                    )
                    target = max(-1.4, min(1.4, target))
                    p.setJointMotorControl2(
                        robot_id,
                        joint_index,
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=self.config.control_force,
                        positionGain=0.28,
                        velocityGain=1.0,
                        physicsClientId=self.client_id,
                    )
                    total_action += abs(target)
                executed_steps += 1

            p.stepSimulation(physicsClientId=self.client_id)
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client_id)
            if base_pos[2] < self.config.fall_height:
                break
        
        final_pos, final_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(final_orn)
        forward_progress = final_pos[0] - start_x
        lateral_drift = abs(final_pos[1] - start_y)
        posture_error = abs(roll) + abs(pitch)
        action_penalty = total_action / max(1, executed_steps)
        survival_fraction = executed_steps / max(1, motion_steps)
        height_bonus = max(0.0, final_pos[2] - self.config.fall_height)
        fallen_penalty = 1.5 if final_pos[2] < self.config.fall_height else 0.0

        fitness = (
            3.0 * forward_progress
            + 1.0 * survival_fraction
            + 0.5 * height_bonus
            - 0.25 * lateral_drift
            - 0.35 * posture_error
            - 0.0008 * action_penalty
            - fallen_penalty
        )
        return (float(fitness),)

    def replay_genome(self, genome: Sequence[float], episode_seconds: float | None = None, settle_seconds: float | None = None, slowdown: float = 1.0, hold: bool = False) -> None:
        """Reproduce un genoma en la GUI actual con pausa y ralentización opcional.

        Esto no devuelve fitness; solo se usa para visualizar.
        """
        ep = episode_seconds if episode_seconds is not None else self.config.episode_seconds
        setsec = settle_seconds if settle_seconds is not None else self.config.settle_seconds

        self._reset_world()
        robot_id = self._load_robot()
        frequency, amplitudes, phases, offsets = self._decode_genome(genome)

        settle_steps = int(setsec / self.config.timestep)
        motion_steps = int(ep / self.config.timestep)
        sleep_time = max(0.0, self.config.timestep * slowdown)

        for step in range(settle_steps + motion_steps):
            if step >= settle_steps:
                t = (step - settle_steps) * self.config.timestep
                for gene_index, joint_index in enumerate(self._joint_indices):
                    target = offsets[gene_index] + amplitudes[gene_index] * math.sin(
                        2.0 * math.pi * frequency * t + phases[gene_index]
                    )
                    target = max(-1.4, min(1.4, target))
                    p.setJointMotorControl2(
                        robot_id,
                        joint_index,
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=self.config.control_force,
                        positionGain=0.28,
                        velocityGain=1.0,
                        physicsClientId=self.client_id,
                    )

            p.stepSimulation(physicsClientId=self.client_id)
            time.sleep(sleep_time)
            base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client_id)
            if base_pos[2] < self.config.fall_height:
                break

        if hold:
            input("Presiona Enter para seguir con las visualizaciones...")
