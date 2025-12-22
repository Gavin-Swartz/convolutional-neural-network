from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytorch_cnn import CNN as torchCNN
from config.read_config import read_config_file


# Load MNIST dataset with PyTorch transformation.
def load_dataset(batch_size, data_dir):
  train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, test_loader


def train_model():
  pass


if __name__ == '__main__':
  # Get configurable data.
  config = read_config_file()

  # Load training and testing sets from dataset.
  train_set, test_set = load_dataset(config['batch_size'], config['data_dir'])

  # Initialize model.
  if config['model'] == 'pytorch':
    model = torchCNN(len(config['classes']))
    print('PyTorch model initialized.')
  else:
    raise NotImplementedError()

  # Train.
  train_model()

  # Save weights
