"""点对点飞行避障环境."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class WindField:
    """简单的风场描述."""

    max_level: int = 3
    base_speed: float = 5.0

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """按照风力等级随机生成一个二维风矢量."""

        level = rng.integers(0, self.max_level + 1)
        speed = level * self.base_speed
        angle = rng.uniform(-math.pi, math.pi)
        return np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)


class PointToPointAvoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        max_steps: int = 500,
        wind_field: Optional[WindField] = None,
        obstacle_centers: Optional[List[Tuple[float, float]]] = None,
        obstacle_radii: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.wind_field = wind_field or WindField()
        # 预设两个圆形障碍物，亦可通过构造参数自定义
        self.obstacle_centers = (
            np.array(obstacle_centers if obstacle_centers is not None else [(4000.0, 3000.0), (7000.0, 8000.0)], dtype=np.float32)
        )
        self.obstacle_radii = np.array(obstacle_radii if obstacle_radii is not None else [1200.0, 1500.0], dtype=np.float32)
        # A点与B点固定为对角位置，便于衡量航迹
        self.start = np.array([0.0, 0.0], dtype=np.float32)
        self.target = np.array([10000.0, 10000.0], dtype=np.float32)
        self.target_radius = 500.0
        self.max_speed = 220.0
        self.min_speed = 80.0
        self.max_turn_rate = math.radians(15.0)
        self.max_acc = 25.0
        self.drag = 0.08
        self.dt = 1.0
        self.safe_distance = 600.0
        self.area_limit = 12000.0

        # 动作为 2 维连续量：油门与转向调整
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 观测包含位置、航向、速度、风向等 11 项特征
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)

        self.np_random = np.random.default_rng(seed)
        self.wind_vector = np.zeros(2, dtype=np.float32)
        self.position = self.start.copy()
        self.heading = 0.0
        self.speed = self.min_speed
        self.steps = 0
        self.prev_distance = self._distance_to_target()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        # 将飞机放置在起点，并沿目标方向初始化航向
        self.position = self.start.copy()
        self.heading = math.atan2(self.target[1] - self.start[1], self.target[0] - self.start[0])
        # 初始速度稍作扰动，提升策略泛化能力
        self.speed = self.min_speed + float(self.np_random.uniform(0.0, 30.0))
        self.steps = 0
        self.wind_vector = self.wind_field.sample(self.np_random)
        self.prev_distance = self._distance_to_target()
        return self._get_obs(), self._get_info(False, False)

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 油门映射到 [0,1]，转向与最大转弯率相乘得到角速度
        throttle = (action[0] + 1.0) / 2.0
        turn = action[1] * self.max_turn_rate

        self.heading += turn
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # 简化动力学模型：油门控制加速度，线性阻力抑制过快
        accel = throttle * self.max_acc - self.drag * self.speed
        self.speed = float(np.clip(self.speed + accel * self.dt, self.min_speed, self.max_speed))
        velocity = np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32) * self.speed
        velocity += self.wind_vector

        self.position = self.position + velocity * self.dt
        self.steps += 1

        distance = self._distance_to_target()
        progress = self.prev_distance - distance
        self.prev_distance = distance

        # 奖励由前进距离驱动，并添加时间惩罚鼓励快速完成任务
        reward = progress * 0.05 - 0.1

        collision = self._check_collision()
        if collision:
            reward -= 80.0

        min_obstacle_distance = self._min_distance_to_obstacles()
        if min_obstacle_distance < self.safe_distance:
            reward -= (self.safe_distance - min_obstacle_distance) * 0.02

        out_of_bounds = np.any(np.abs(self.position) > self.area_limit)
        if out_of_bounds:
            reward -= 40.0

        reached = distance <= self.target_radius
        if reached:
            reward += 200.0 + max(0.0, (self.max_steps - self.steps) * 0.2)

        terminated = reached or collision or out_of_bounds
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = self._get_info(reached, collision)

        return obs, reward, terminated, truncated, info

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self.target - self.position))

    def _distance_to_obstacle(self, idx: int) -> float:
        center = self.obstacle_centers[idx]
        return float(np.linalg.norm(center - self.position) - self.obstacle_radii[idx])

    def _min_distance_to_obstacles(self) -> float:
        distances = [self._distance_to_obstacle(i) for i in range(len(self.obstacle_centers))]
        return float(min(distances))

    def _check_collision(self) -> bool:
        distances = [self._distance_to_obstacle(i) for i in range(len(self.obstacle_centers))]
        return any(dist <= 0 for dist in distances)

    def _get_obs(self) -> np.ndarray:
        rel_target = self.target - self.position
        heading_to_target = math.atan2(rel_target[1], rel_target[0])
        heading_error = (heading_to_target - self.heading + math.pi) % (2 * math.pi) - math.pi

        normalized_pos = self.position / self.area_limit
        normalized_speed = (self.speed - self.min_speed) / (self.max_speed - self.min_speed)
        normalized_distance = self.prev_distance / np.linalg.norm(self.target - self.start)
        obstacle_dists = [self._distance_to_obstacle(i) / self.area_limit for i in range(len(self.obstacle_centers))]
        min_obstacle_distance = min(obstacle_dists)
        wind_norm = np.clip(self.wind_vector / (self.wind_field.max_level * self.wind_field.base_speed + 1e-6), -1.0, 1.0)

        obs = np.array(
            [
                normalized_pos[0],
                normalized_pos[1],
                math.cos(self.heading),
                math.sin(self.heading),
                normalized_speed * 2.0 - 1.0,
                math.cos(heading_error),
                math.sin(heading_error),
                normalized_distance * 2.0 - 1.0,
                min_obstacle_distance * 2.0,
                wind_norm[0],
                wind_norm[1],
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self, reached: bool, collision: bool) -> Dict:
        return {
            "is_success": reached,
            "collision": collision,
            "wind": self.wind_vector.copy(),
            "steps": self.steps,
            "position": self.position.copy(),
        }

    def render(self):
        return None

