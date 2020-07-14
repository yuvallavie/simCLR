# --------------------------------------------------------------------------------------------- #
                                            # Abstract
# --------------------------------------------------------------------------------------------- #

'''
simCLR linear evaluation using pre-trained weights by https://github.com/google-research/simclr
- ResNet50x1 https://drive.google.com/file/d/13x2-QBIF1s6EkTWf1AjHEGUc4v047QVF/view?usp=sharing
  Please download this file and extract its contents in the "checkpoints" folder.

this file is added as a resource to my seminar presentation at Bar-Ilan University.
Author : Yuval Lavie

File locations:
    - ../data
    - ../models
    - ../checkpoints
'''
#%%
# Imports
# Utilities
import numpy as np;
import sys;
import os;

# Allows me to structure the folders and import
sys.path.append('./models')

# Torch CORE
import torch;
import torch.nn.functional as F;
import torch.optim as optim;
from torch.optim import lr_scheduler;



#%%
# --------------------------------------------------------------------------------------------- #
                                            # Description
# --------------------------------------------------------------------------------------------- #
'''
We will compare the results of fine-tuning a linear classifier on top of the representations learned by simCLR and by a supervised resnet50 pretrained from pytorch.

The quality of the representations is measured by multiclass logistic regression.

- Unsupervised corpus (100k)
- Training set (4k)
- Validation set (1k)
- Test set (8k).
 '''


#%%
# Make sure that the model file has been downloaded before starting
downloaded = os.path.isfile('./checkpoints/resnet50-1x.pth');
if(downloaded == False):
    print("You have not downloaded the checkpoints, please refer to the readme file at the checkpoints directory..");
    exit(1);

print("Checkpoint exists, continuing..");

#%%
from Data_Utilities import Get_SupervisedLoaders;
print("------------------------------------------------------------------")
print("                       Downloading the data                       ")
print("------------------------------------------------------------------")
# Get the STL-10 Data set
test_loader , train_val_loaders = Get_SupervisedLoaders(batch_size = 256,validation_size = 0.2,root='./data');

# Define the class names
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

#%%
# Visualize the some samples from the data set
print("------------------------------------------------------------------")
print("                       Visualzing the data                        ")
print("------------------------------------------------------------------")
import matplotlib.pyplot as plt;

examples = enumerate(test_loader);
batch_idx, (example_data, example_targets) = next(examples);

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title(classes[example_targets[i]])
  plt.xticks([])
  plt.yticks([])
plt.show()


#%%
print("------------------------------------------------------------------")
print("                       Loading the model                          ")
print("------------------------------------------------------------------")

# Select the device to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

print("Loading google's simCLR model")
from simCLR_model import SimCLR;
unsupervised_model = SimCLR(device);

#%%
# --------------------------------------------------------------------------------------------- #
                                # Finetuning the evaluator model
# --------------------------------------------------------------------------------------------- #

print("------------------------------------------------------------------")
print("         Finetuning the linear evaluator on top of simCLR         ")
print("------------------------------------------------------------------")

# Evaluate with a linear head
from linear_model import Evaluator;

evaluator = Evaluator(device,unsupervised_model.encoder)

# Set the network to CUDA
evaluator.to(device);


# Set the optimizer
optimizer = optim.Adam(evaluator.parameters(),lr=0.0001);

# Set the criterion
criterion = F.nll_loss;

# Set the schedueler
# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Set the network to training mode
evaluator.train();

print("Fine-tuning the linear evaluator model");
# Train the network
evaluator.fit(train_val_loaders, criterion, optimizer, scheduler, num_epochs=20);


# Get the results on the test data
print("Calculating the accuracy of the linear evalulator on the test set.");
unsup_acc = evaluator.calculateAccuracy(test_loader);

#%%
from supervised import Supervised_RL;
supervised_model = Supervised_RL(device);

print("------------------------------------------------------------------")
print(" Finetuning the linear evaluator on top of a pretrained resnet50  ")
print("------------------------------------------------------------------")

# Evaluate with a linear head
from linear_model import Evaluator;

evaluator = Evaluator(device,supervised_model.encoder);

# Set the network to CUDA
evaluator.to(device);

# Print the network
#print(evaluator)

# Set the optimizer
optimizer = optim.Adam(evaluator.parameters(),lr=0.0001);

# Set the criterion
criterion = F.nll_loss;

# Set the schedueler
# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1);

# Set the network to training mode
evaluator.train();

print("Fine-tuning the linear evaluator model")
# Train the network
evaluator.fit(train_val_loaders, criterion, optimizer, scheduler, num_epochs=20);


# Get the results on the test data
print("Calculating the accuracy of the linear evalulator on the test set.");
sup_acc = evaluator.calculateAccuracy(test_loader);

#%%

import matplotlib.pyplot as plt;

height = [unsup_acc, sup_acc]
bars = ('simCLR', 'supervised')
y_pos = np.arange(len(bars))

plt.bar(y_pos, height, color=['red', 'cyan'])
plt.xticks(y_pos, bars)
plt.xlabel("Models")
plt.ylabel("Finetuned accuracy")
plt.title("Finetuned accuracy of simCLR and ResNet50")
plt.show()

