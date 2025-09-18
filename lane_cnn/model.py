import torch
import torch.nn as nn
import torch.nn.functional as F

class LaneCNN(nn.Module):
    def __init__(self):
        super(LaneCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        # We'll determine this dynamically later
        self.flatten = nn.Flatten()
        self._fc_input_dim = None

        # Placeholder linear layers (will reinitialize once we know shape)
        self.fc1 = None
        self.fc2 = nn.Linear(512, 1)  # Final output is scalar steering

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)

        # Initialize fc1 dynamically
        if self.fc1 is None:
            self._fc_input_dim = x.size(1)
            self.fc1 = nn.Linear(self._fc_input_dim, 512).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
