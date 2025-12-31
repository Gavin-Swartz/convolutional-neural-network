import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 8, 3)        # Create 2D convolutional layer given a single input channel. 8 filters and 3x3 kernel.
    self.pool = nn.MaxPool2d(2, 2)          # Max pooling layer with 2x2 kernel size and stride of 2.
    self.conv2 = nn.Conv2d(8, 16, 3)       # Use output channels of conv1 (8) to create 16 feature maps using 3x3 kernel.
    self.fc1 = nn.Linear(16 * 5 * 5, 120)   # Fully connected layer with 16 feature maps * kernel (5x5) to get output of 120 neurons.
    self.fc2 = nn.Linear(120, 84)           # Second FC layer given fc1 neurons as input to get 84 output neurons.
    self.fc3 = nn.Linear(84, num_classes)   # Third FC layer given fc2 neurons as input to get 1 output per class.

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))    # Apply convolution, ReLu activation, and downsample feature maps using max pooling.
    x = self.pool(F.relu(self.conv2(x)))    # Apply second convolution.
    x = torch.flatten(x, 1)                 # Flatten the tensor.
    x = F.relu(self.fc1(x))                 # Apply fully connected layer for learning abstract feature combinations.
    x = F.relu(self.fc2(x))                 # Apply another FC layer for further abstraction.
    x = self.fc3(x)                         # Output layer produces logits.
    return x
