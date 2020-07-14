# Simple classification model
import torch
import torch.nn as nn;


# Utilities
import numpy as np;
import time;
import copy;
#%%
class Evaluator(nn.Module):
    def __init__(self,device,encoder):
        super(Evaluator, self).__init__()
        self.device = device;
        self.encoder = encoder;

        Linear = [
                  nn.Linear(2048,10),
                  nn.Softmax(dim=1)
                  ];

        self.fc = nn.Sequential(*Linear);

    def forward(self,x):
        x = self.encoder(x);
        return self.fc(x);

    def fit(self,dataloaders, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        model = self;
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_epoch = 0;

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best validation accuracy:");
        print(f"Epoch: {best_epoch} , Accuracy: {best_acc}")
        #print('Best validation Accurracy {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def predict_batch(self,data):
        predictions = [];
        with torch.no_grad():
            for sample in data:
                data = sample[0].to(self.device);
                val = self.forward(data.reshape(1,3,96,96));
                unused,val = val.max(1)
                predictions.append(val.item());
        return np.array(predictions);

    def predict_single(self,sample):
        with torch.no_grad():
            sample = torch.FloatTensor(sample);
            sample = sample.to(self.device);
            val = self.forward(sample.reshape(1,3,96,96));
            proba,val = val.max(1);
            return proba.item(),val.item();

    def calculateAccuracy(self,Loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in Loader:
                images, labels = data[0].to(self.device),data[1].to(self.device)
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = (100 * correct / total);
        print('Accuracy of the network %d %%' %acc)
        return acc;