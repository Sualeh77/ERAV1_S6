import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from utils import GetCorrectPredCount
import matplotlib.pyplot as plt

########################################################################################################################################################

########################################################################  Block 1   #########################################################################
##################################################################### Model Basic setup ##########################################################################

class Model_1_1(nn.Module):
    """
        Initial Model Setup, Passing device for using mps in mac.
    """
    def __init__(self, device):
        super(Model_1_1, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 512, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(512, 1024, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 |
        self.conv7 = nn.Conv2d(1024, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = x.to(self.device)
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

############################################################# Setting up Model Skeleton #################################################################

class Model_1_2(nn.Module):
    """
        Model to setup skeleton.
    """
    def __init__(self, device):
        super(Model_1_2, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0),                  # 1x28x28 > 16x26x26 | 3
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=0),                  # 16x26x26 > 32x24x24 | 5
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=0),                  # 32x24x24 > 64x22x22 | 7
            nn.ReLU()
        )
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                # 64x22x22 > 64x11x11 | 7
            nn.Conv2d(64, 16, kernel_size=1)                   # 64x11x11 > 16x11x11 |7 | j = 2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=0),                    # 16x11x11 > 32x9x9 | 11
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=0),                     # 32x9x9 > 64x7x7 | 15
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=0),                    # 64x7x7 > 128x5x5 | 19
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=0)                    # 128x5x5 > 128x3x3 | 23
        )
        #Stopping here as it seems 23 RF must cover number present in image.
        self.op = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1),               # 128x3x3 > 10x3x3
            nn.AvgPool2d(3)                                  # 10x3x3 > 10x1x1
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.op(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

######################################################### Making MODEL lighter with under 8k params #########################################################

class Model_1_3(nn.Module):
    """
        reducing params below 8k, with leaving buffer for batch norm.
    """
    def __init__(self, device):
        super(Model_1_3, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=0),                  # 1x28x28 > 10x26x26 | 3
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0),                  # 10x26x26 > 10x24x24 | 5
            nn.ReLU(),
            nn.Conv2d(10, 16, 3, padding=0),                  # 10x24x24 > 16x22x22 | 7
            nn.ReLU()
        )
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                # 16x22x22 > 16x11x11 | 7
            nn.Conv2d(16, 10, kernel_size=1)                   # 16x11x11 > 10x11x11 |7 | j = 2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0),                    # 10x11x11 > 10x9x9 | 11
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0),                     # 10x9x9 > 10x7x7 | 15
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0),                    # 10x7x7 > 10x5x5 | 19
            nn.ReLU(),
            nn.Conv2d(10, 22, 3, padding=0)                    # 10x5x5 > 22x3x3 | 23
        )
        #Stopping here as it seems 23 RF must cover number present in image. Leaving some room for Batch Norm params.
        self.op = nn.Sequential(
            nn.Conv2d(22, 10, kernel_size=1),               # 22x3x3 > 10x3x3
            nn.AvgPool2d(3)                                  # 10x3x3 > 10x1x1
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.op(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

########################################################################################################################################################

########################################################################  Block 2   #########################################################################
##########################################################################################################################################################
#Appying Batch Norm | Also changed model structure to distribute weight after checking accuracies of batch norm

class Model_2_1(nn.Module):
    """
        Applying Batch Norm. Also changed model structure to distribute weight after checking accuracies of batch norm
    """
    def __init__(self, device):
        super(Model_2_1, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=0),                  # 1x28x28 > 10x26x26 | 3
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 3, padding=0),                  # 10x26x26 > 10x24x24 | 5
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 20, 3, padding=0),                  # 10x24x24 > 20x22x22 | 7
            nn.ReLU(),
            nn.BatchNorm2d(20)
        )
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                # 20x22x22 > 20x11x11 | 7
            nn.Conv2d(20, 10, kernel_size=1),                   # 20x11x11 > 10x11x11 |7 | j = 2
            nn.BatchNorm2d(10)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0),                    # 10x11x11 > 10x9x9 | 11
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 3, padding=0),                     # 10x9x9 > 10x7x7 | 15
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 3, padding=0),                    # 10x7x7 > 10x5x5 | 19
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 20, 3, padding=0)                    # 10x5x5 > 20x3x3 | 23
        )
        #Stopping here as it seems 23 RF must cover number present in image. Leaving some room for Batch Norm params.
        self.op = nn.Sequential(
            nn.Conv2d(20, 10, kernel_size=1),               # 20x3x3 > 10x3x3
            nn.AdaptiveAvgPool2d((1,1))                                  # 10x3x3 > 10x1x1
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.op(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

########################################################################  Block 3   #########################################################################
# In block 3 only changed .pynb file to apply image augmentation and Step LR.
##########################################################################################################################################################
    

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