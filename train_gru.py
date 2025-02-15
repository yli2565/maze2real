import wandb
import torch
from agent.gru import GRUAgent
from env.memory_maze.tasks import memory_maze_9x9

# Training config
config = {
    "total_steps": 100_000,
    "batch_size": 512,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "hidden_size": 256,
    "num_layers": 2,
    "eval_freq": 10_000
}

def train():
    # Initialize environment
    env = memory_maze_9x9(
        discrete_actions=True,
        image_only_obs=False,
        global_observables=True,
        camera_resolution=64
    )
    
    # Initialize agent
    agent = GRUAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        learning_rate=config["learning_rate"]
    )
    
    # Initialize WandB
    wandb.init(project="memory-maze", config=config)
    
    # Training loop
    obs = env.reset()
    hidden_state = None
    total_reward = 0
    
    for step in range(config["total_steps"]):
        # Get action from agent
        action, hidden_state = agent.get_action(obs, hidden_state)
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)
        
        # Update
        if len(agent.buffer) >= config["batch_size"]:
            loss = agent.update()
            wandb.log({"loss": loss})
        
        # Logging
        total_reward += reward
        if done:
            wandb.log({"episode_reward": total_reward})
            total_reward = 0
            hidden_state = None
            obs = env.reset()
        else:
            obs = next_obs
            
        # Evaluation
        if step % config["eval_freq"] == 0:
            avg_reward = evaluate(agent, env)
            wandb.log({"eval_reward": avg_reward})
            torch.save(agent.state_dict(), f"gru_agent_{step}.pt")

def evaluate(agent, env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()
        hidden_state = None
        episode_reward = 0
        
        while True:
            action, hidden_state = agent.get_action(obs, hidden_state)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                total_reward += episode_reward
                break
                
    return total_reward / num_episodes

if __name__ == "__main__":
    train()
