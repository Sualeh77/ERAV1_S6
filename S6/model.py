import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from utils import GetCorrectPredCount
import matplotlib.pyplot as plt

########################################################################################################################################################
class Net(nn.Module):
    """
     Accuracy 99.59 | step 12 | params ~19k | batch 128
    """

    def __init__(self, device):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),                                                # 1x28x28 > 16x28x28 | RF 3
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),                                               # 16x28x28 > 16x28x28 | RF 5
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),                                               # 16x28x28 > 16x28x28 | RF 7
            nn.ReLU()
        )
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                                                         # 16x28x28 > 16x14x14 | RF 8 | J 2
            #nn.Conv2d(32, 16, kernel_size=1)  # Removed as it was reducing acc from 99.59 to 99.53     # 32x14x14 > 16x14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),                                               # 16x14x14 > 16x12x12 | RF 12
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),                                               # 16x12x12 > 16x10x10 | RF 16
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),                                               # 16x10x10 > 16x8x8 | RF 20
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),                                               # 16x8x8 > 16x6x6 | RF 24
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),                                               # 16x6x6 > 32x4x4 | RF 28
        )    
        self.antman = nn.Conv2d(32, 10, 1)                                                # 32x4x4 > 10x4x4
        self.gap = nn.AvgPool2d(4)
        

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.antman(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
########################################################################################################################################################


########################################################################################################################################################
def model_summary(model, input_size=(1, 28, 28)):
    return summary(model, input_size)
########################################################################################################################################################


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


########################################################################################################################################################
def model_train(model, device, train_loader, optimizer, criterion):
    """
        Training method
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate Loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
########################################################################################################################################################


########################################################################################################################################################
def model_test(model, device, test_loader, criterion):
    """
        Test method.
    """
    model.eval()
    
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            
            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
#########################################################################################################################################################

#########################################################################################################################################################
def draw_graphs():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
#########################################################################################################################################################