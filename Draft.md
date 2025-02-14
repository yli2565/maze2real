# Maze Partially Observable Sim2Real

## Sotry

- Relation with football

## Project structure

.
├── Draft.md
├── env
│   ├── gui
│   │   ├── recording.py
│   │   ├── requirements.txt
│   │   └── run_gui.py
│   ├── LICENSE
│   ├── memory_maze
│   │   ├── gym_wrappers.py
│   │   ├── helpers.py
│   │   ├── __init__.py
│   │   ├── maze.py
│   │   ├── oracle.py
│   │   ├── tasks.py
│   │   └── wrappers.py
│   ├── README.md
│   └── setup.py
├── logs
├── networks
├── references
│   └── Evaluating Long-Term Memory in 3D Mazes.pdf
├── requirements.txt
└── scripts

## Architecture

First just try GRU

## Coding

env: git@github.com:yli2565/memory-maze.git, use it as a base env, we might need to modify it. I've already add it as submodule in this project

logging: Weights & Biases

network: PyTorch, seperate from the agent,implement each network in a different file, we might need to modify it frequently

callback: Wandb callback, tensorboard callback, eval & save model callback

