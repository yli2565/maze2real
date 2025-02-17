# Maze Partially Observable Sim2Real

## Sotry

- Relation with football

## Project structure

.
├── agent (folder)
├── Draft.md
├── env (folder)
│   git sub module
├── logs
├── memory_maze -> env/memory_maze
├── networks (folder)
├── references
│   └── Evaluating Long-Term Memory in 3D Mazes.pdf
├── requirements.txt
├── requirements.yml
├── scripts (folder)
├── train_impala.py (draftup but not working)
## Environment

The Memory Maze environment is implemented using MuJoCo (Todorov et al., 2012) as the physics
and graphics engine and the dm_control (Tunyasuvunakool et al., 2020) library for building RL
environments. The environment can be installed as a pip package memory-maze or from the source
code, available on the project website . There are four Memory Maze tasks with varying sizes and
difficulty: Memory 9x9, Memory 11x11, Memory 13x13, and Memory 15x1

Code to create env 

from memory_maze.gym_wrappers import GymWrapper
from memory_maze.tasks import memory_maze_9x9, memory_maze_11x11, memory_maze_13x13, memory_maze_15x15

def create_env(size='9x9', seed=None):
    """
    Create a Memory Maze environment.
    
    Args:
        size: One of '9x9', '11x11', '13x13', '15x15'
        seed: Random seed for environment
    
    Returns:
        GymWrapper: A gym-compatible environment
    """
    maze_creators = {
        '9x9': memory_maze_9x9,
        '11x11': memory_maze_11x11, 
        '13x13': memory_maze_13x13,
        '15x15': memory_maze_15x15
    }
    
    if size not in maze_creators:
        raise ValueError(f"Size must be one of {list(maze_creators.keys())}")
        
    env = maze_creators[size](seed=seed)
    return GymWrapper(env)

## Architecture
### Training
For Policy Implementation:

Action Space:

The environment has 6 discrete actions (from Table F.1):

Copy0: noop [0.0, 0.0]
1: forward [+1.0, 0.0]
2: turn left [0.0, -1.0]
3: turn right [0.0, +1.0]
4: forward left [+1.0, -1.0]
5: forward right [+1.0, +1.0]

DreamerV2 Policy:

Uses an actor-critic architecture
Actor generates continuous actions that are discretized
Policy is trained through imagined rollouts using the world model
Key components:

World Model: RSSM (2048 recurrent units)
Actor MLP: Used for policy
Critic MLP: Value estimation

Training details:

Imagination horizon: 15 steps
Uses λ-returns for advantage estimation
Actor entropy loss scale: 0.001
Slow critic updates every 100 steps

IMPALA Policy:

Standard actor-critic setup with V-trace
LSTM with 256 units for memory
Uses V-trace importance weighting for off-policy correction
Parallel training with 128 actors
### Evaluation
For Evaluation:

Training Metrics:

Trained for 100M environment steps
Average return over episodes
They ran 5 different random seeds for each configuration
Additional 400M steps for IMPALA to show learning curve

### OFFLINE PROBING
We don't need this functionality now, but please leave some room for it. For example, the policy netwokr should be seperated from agent and could be easily imported to train on this offline dataset

Unsupervised representation learning aims to learn representations that can later be used for downstream tasks of interest. In the context of partially observable environments, we would like unsupervised representations to summarize the history of observations into a representation that contains
information about the state of the environment beyond what is visible in the current observation by
remembering salient information about the environment. Unsupervised representations are commonly
evaluated by probing (Oord et al., 2018; Chen et al., 2020; Gregor et al., 2019; Anand et al., 2019),
where a separate network is trained to predict relevant properties from the frozen representations.
We introduce the following four Memory Maze offline probing benchmarks: Memory 9x9 Walls,
Memory 15x15 Walls, Memory 9x9 Objects, and Memory 15x15 Objects. These are based on either
using the maze wall layout (maze_layout) or agent-centric object locations (targets_vec) as
the probe prediction target, trained and evaluated on either Memory Maze 9x9 (30M) or Memory
Maze 15x15 (30M) offline datasets.

## Coding

env: git@github.com:yli2565/memory-maze.git, use it as a base env, we might need to modify it. I've already add it as submodule in this project

conda env name: maze2real, I've already installed all required packages, you can directly use it, it has sb3, ray, pytorch, numpy al

logging: Weights & Biases 

For Wandb, here's the doc:
Ray Tune
How to integrate W&B with Ray Tune.
  2 minute read  

W&B integrates with Ray by offering two lightweight integrations.

TheWandbLoggerCallback function automatically logs metrics reported to Tune to the Wandb API.
The setup_wandb() function, which can be used with the function API, automatically initializes the Wandb API with Tune’s training information. You can use the Wandb API as usual. such as by using wandb.log() to log your training process.
Configure the integration
from ray.air.integrations.wandb import WandbLoggerCallback
Python
Wandb configuration is done by passing a wandb key to the config parameter of tune.run() (see example below).

The content of the wandb config entry is passed to wandb.init() as keyword arguments. The exception are the following settings, which are used to configure the WandbLoggerCallback itself:

Parameters
project (str): Name of the Wandb project. Mandatory.

api_key_file (str): Path to file containing the Wandb API KEY.

api_key (str): Wandb API Key. Alternative to setting api_key_file.

excludes (list): List of metrics to exclude from the log.

log_config (bool): Whether to log the config parameter of the results dictionary. Defaults to False.

upload_checkpoints (bool): If True, model checkpoints are uploaded as artifacts. Defaults to False.

Example
from ray import tune, train
from ray.air.integrations.wandb import WandbLoggerCallback


def train_fc(config):
    for i in range(10):
        train.report({"mean_accuracy": (i + config["alpha"]) / 10})


tuner = tune.Tuner(
    train_fc,
    param_space={
        "alpha": tune.grid_search([0.1, 0.2, 0.3]),
        "beta": tune.uniform(0.5, 1.0),
    },
    run_config=train.RunConfig(
        callbacks=[
            WandbLoggerCallback(
                project="<your-project>", api_key="<your-api-key>", log_config=True
            )
        ]
    ),
)

results = tuner.fit()
Python
setup_wandb
from ray.air.integrations.wandb import setup_wandb
Python
This utility function helps initialize Wandb for use with Ray Tune. For basic usage, call setup_wandb() in your training function:

from ray.air.integrations.wandb import setup_wandb


def train_fn(config):
    # Initialize wandb
    wandb = setup_wandb(config)

    for i in range(10):
        loss = config["a"] + config["b"]
        wandb.log({"loss": loss})
        tune.report(loss=loss)


tuner = tune.Tuner(
    train_fn,
    param_space={
        # define search space here
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb configuration
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
Python


network: PyTorch, seperate from the agent,implement each network in a different file, we might need to modify it frequently. For this, let's first try IMPALA (LSTM)

callback: eval & save model callback (skip for now)

task1(done): First, try to use the gym env + ray's implementation of IMPALA https://github.com/ray-project/ray/blob/master/rllib/algorithms/impala/impala.py

task2: Debug the code with conda run -n maze2real WANDB_API_KEY=653b4ba04cb6020806abb0ec7b01dadf74277454 python train_impala.py

