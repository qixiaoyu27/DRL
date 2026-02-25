from drl_uav.envs.coverage_env import FixedWingCoverageEnv, EnvConfig


def main():
    env = FixedWingCoverageEnv(EnvConfig())
    obs, _ = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
    print("smoke test passed", obs.shape, reward, info["coverage"])


if __name__ == "__main__":
    main()
