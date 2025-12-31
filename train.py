import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from pytorch_cnn import CNN as torchCNN
from config.read_config import read_config_file


# Load MNIST dataset with PyTorch transformation.
def load_dataset(batch_size, data_dir):
  train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, test_loader


def train_model(epochs, train_loader, model, learning_rate):
  # Loss and optimizer.
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(epochs):
    print(f'Starting epoch {epoch}')
    for i, data in enumerate(train_loader, 0):
      # Get inputs and labels from data.
      inputs, labels = data

      # Forward propagation and compute loss.
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      # Reset gradients and perform backpropagation.
      optimizer.zero_grad()
      loss.backward()

      # Update weights.
      optimizer.step()


def eval(data, device, model):
  num_correct = 0
  num_samples = 0
  model.eval()

  # Evaluate without computing gradients.
  with torch.no_grad():
    print('Starting model evaluation.')
    for x, y in data:
      x = x.to(device=device)
      y = y.to(device=device)

      scores = model(x)
      _, predictions = scores.max(1)

      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)

    print(f'{float(num_correct)/float(num_samples)*100: .2f}% Accuracy')


if __name__ == '__main__':
  # Get configurable data.
  config = read_config_file()

  # Set device.
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Load training and testing sets from dataset.
  train_loader, test_loader = load_dataset(config['batch_size'], config['data_dir'])

  # Initialize the model.
  if config['model'] == 'pytorch':
    model = torchCNN(len(config['classes']))
    print('PyTorch model initialized.')
  else:
    raise NotImplementedError()

  # Train the model.
  train_model(config['epochs'], train_loader, model, config['learning_rate'])

  # Evaluate the model.
  eval(test_loader, device, model)