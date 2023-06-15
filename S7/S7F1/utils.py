import matplotlib.pyplot as plt
import math

#########################################################################################################################################################
def return_dataset_images(train_loader, total_images):
    """
        This function prints images from train loader.
        Params : {train_loader : training data loader, total_images : no. of images to be printed}
    """
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(total_images):
        #plt.subplot(3,4,i+1)
        plt.subplot(4,math.ceil(total_images/4),i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
#########################################################################################################################################################


#########################################################################################################################################################
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
#########################################################################################################################################################
