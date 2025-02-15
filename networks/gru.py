import torch
import torch.nn as nn

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[:, -1, :])  # Take last time step's output
        return out, hidden
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
