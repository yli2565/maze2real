from abc import ABC, abstractmethod
import torch
from typing import Any, Dict

class BaseAgent(ABC):
    """Base class for reinforcement learning agents"""
    
    def __init__(self, network: torch.nn.Module, device: torch.device):
        self.network = network
        self.device = device
        
    @abstractmethod
    def act(self, state: Any) -> Any:
        """Select action based on current state"""
        pass
        
    @abstractmethod
    def remember(self, experience: Dict[str, Any]):
        """Store experience in memory buffer"""
        pass
        
    @abstractmethod
    def learn(self) -> Dict[str, float]:
        """Update network weights based on experiences"""
        pass
        
    def save_checkpoint(self, path: str):
        """Save agent state to file"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load agent state from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
