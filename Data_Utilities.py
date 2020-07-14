# Imports
# This section contains all of our imports including the core Torch library and utilities
# Utilities

from torch.utils.data import random_split;
from torch.utils.data import DataLoader;

# Torch Vision
import torchvision.datasets as datasets;
import torchvision.transforms as transforms

# Transformations applied on the data.
transform = transforms.Compose(
    [transforms.ToTensor()])

#%%
def Get_SupervisedLoaders(batch_size,validation_size,root):

    print("Downloading the training data..")
    # Download the STL-10 training set
    train_validation_data = datasets.STL10(root=root, split='train',
                                            download=True,transform = transform);
    # Verify its size
    print(f"Training data size is: {len(train_validation_data)}");


    # Calculate the sizes of train and validation
    lengths = [int(len(train_validation_data)*(1-validation_size)), int(len(train_validation_data)*validation_size)];

    print("Splitting to train and validation..")
    # Split to training and validation
    training_data,validation_data = random_split(train_validation_data, lengths);

    print(f"Training data size is: {len(training_data)} Validation data size is: {len(validation_data)}");

    print("Creating the loaders..")
    # Create the loaders
    train_loader = DataLoader(training_data, batch_size=batch_size,
                                              shuffle=True, num_workers=0);
    validation_loader = DataLoader(validation_data, batch_size=batch_size,
                                              shuffle=True, num_workers=0);

    print("Downloading the test data..")
    # Download the STL-10 test set
    test_data = datasets.STL10(root=root, split='test',
                                            download=True,transform = transform);

    # Verify its size
    print(f"Test data size is: {len(test_data)}");

    test_loader = DataLoader(test_data, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    print("Returned test_loader and a dataloaders hashmap with train and val as keys");

    return test_loader,{"train":train_loader,"val":validation_loader};

def Get_UnsupervisedLoader(batch_size,validation_size,root):

    print("Downloading the unsupervised data..")
    # Download the STL-10 unsupervised set
    data = datasets.STL10(root=root, split='unlabeled',
                                            download=True,transform = transform);
    # Verify its size
    print(f"Training data size is: {len(data)}");

    loader = DataLoader(data, batch_size=batch_size,
                                         shuffle=True, num_workers=0);
    return loader;
