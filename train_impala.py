import numpy as np
import ray
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import ppo_catalog
from gymnasium_wrappers import GymnasiumWrapper
from memory_maze.tasks import (
    memory_maze_9x9,
    memory_maze_11x11,
    memory_maze_13x13,
    memory_maze_15x15,
)


def create_env(config):
    """
    Create a Memory Maze environment.
    """
    maze_creators = {
        "9x9": memory_maze_9x9,
        "11x11": memory_maze_11x11,
        "13x13": memory_maze_13x13,
        "15x15": memory_maze_15x15,
    }

    size = config.get("size", "9x9")
    seed = config.get("seed", None)

    if size not in maze_creators:
        raise ValueError(f"Size must be one of {list(maze_creators.keys())}")

    env = maze_creators[size](seed=seed)
    return GymnasiumWrapper(env)


def main():
    # Create a sample environment to get observation and action spaces
    sample_env = create_env({"size": "9x9"})
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    total_obs_size = sum(np.prod(space.shape) for space in obs_space.spaces.values())

    # Then initialize Ray
    ray.init(
        runtime_env={
            "env_vars": {"RAY_DEBUG_POST_MORTEM": "1"},
        },
    )
    # First, register the environment
    register_env("MemoryMaze-v0", create_env)
    # Calculate total observation size

    # Configure IMPALA algorithm
    config = (
        ImpalaConfig()
        .environment(
            env="MemoryMaze-v0",  # Use the registered environment name
            env_config={"size": "9x9"},
        )
        .training(
            train_batch_size=512,
            lr=0.0005,
            grad_clip=40.0,
            model={
                "use_lstm": True,
                "lstm_cell_size": 256,
                "lstm_use_prev_action": False,
                "lstm_use_prev_reward": False,
                # Set input size matching the flattened observation space
                "fcnet_hiddens": [512],
                # "conv_filters": [
                #     [32, 4, 2],  # Conv2d(3 -> 32, kernel=4, stride=2)
                #     [64, 4, 2],  # Conv2d(32 -> 64, kernel=4, stride=2)
                #     [128, 4, 2],  # Conv2d(64 -> 128, kernel=4, stride=2)
                #     [256, 4, 2],  # Conv2d(128 -> 256, kernel=4, stride=2)
                # ],
            },
        )
        .resources(num_gpus=1)
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=2,
            remote_worker_envs=True,
            rollout_fragment_length=50,
        )
        .framework("torch")
        .evaluation(evaluation_interval=1000)
        # Disable the new API stack
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
    )

    sample_env.close()

    # Configure WandB integration
    wandb_config = WandbLoggerCallback(
        project="maze2real",
        api_key="653b4ba04cb6020806abb0ec7b01dadf74277454",
        log_config=True,
    )

    # Create tuner
    tuner = tune.Tuner(
        "IMPALA",
        param_space=config,
        run_config=train.RunConfig(
            callbacks=[wandb_config],
            stop={"training_iteration": 100},
            checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
        ),
    )

    # Execute training
    results = tuner.fit()


def debugEnv():
    envInstance = create_env({"size": "9x9"})
    obs, _ = envInstance.reset()
    print("Observation keys and shapes:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
    totalObsSize = sum(
        np.prod(value.shape) for value in obs.values() if isinstance(value, np.ndarray)
    )
    print("Total observation size:", totalObsSize)


if __name__ == "__main__":
    main()
