import torch
from typing import Any, Dict
from .base import BaseAgent
from collections import deque
import random

class GRUAgent(BaseAgent):
    """GRU-based reinforcement learning agent with memory"""
    
    def __init__(self, 
                 network: torch.nn.Module, 
                 device: torch.device,
                 hidden_size: int, 
                 buffer_size: int = 10000, 
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99) -> None:
        super().__init__(network, device)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.hidden = None  # Single initialization
        
    def reset_hidden(self):
        """Reset hidden state between episodes"""
        self.hidden = torch.zeros(1, self.hidden_size).to(self.device)
        
    def act(self, state: torch.Tensor) -> int:
        """Select action using current policy and hidden state"""
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action_probs, self.hidden = self.network(state, self.hidden)
            return torch.argmax(action_probs).item()
            
    def remember(self, experience: Dict[str, Any]):
        """Store experience in replay buffer"""
        self.memory.append(experience)
        if experience['done']:
            self.reset_hidden()
        
    def learn(self) -> Dict[str, float]:
        """Update network weights using experiences from memory"""
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([exp['state'] for exp in batch]).to(self.device)
        actions = torch.tensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.tensor([exp['done'] for exp in batch]).to(self.device)
        
        # Initialize hidden states
        hidden = torch.zeros(1, self.hidden_size).to(self.device)
        
        # Forward pass
        q_values, _ = self.network(states, hidden)
        current_q = q_values.gather(1, actions.unsqueeze(1))
        
        # Calculate target
        with torch.no_grad():
            next_q_values, _ = self.network(next_states, hidden)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (
                (1 - dones) * self.discount_factor * max_next_q
            )
            
        # Calculate loss
        loss = torch.nn.functional.mse_loss(current_q.squeeze(), target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
