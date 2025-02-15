import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNetwork(nn.Module, ABC):
    """Base class for all neural networks in the project"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation"""
        pass
        
    def save(self, path: str):
        """Save model weights to specified path"""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        """Load model weights from specified path"""
        self.load_state_dict(torch.load(path, map_location=self.device))
