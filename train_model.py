#%%
from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_network_param(action, sensor_readings):
    network_param = np.append(sensor_readings, [action, 0])
    network_param = network_param.flatten()[:-1]
    network_param = torch.as_tensor(network_param, dtype=torch.float32)
    return network_param    

def train_model(no_epochs):
    batch_size = 50
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    losses = []
    
    #min_loss = model.evaluate(model, data_loaders.test_loader, criterion)
    #losses.append(min_loss)
    criterion = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    prediction = []
    label = []
    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader):
            optimizer.zero_grad()
            output = model(sample['input'])

            loss = criterion(output.reshape(-1), sample['label'])
            loss.backward()
            optimizer.step()
       
        model.eval()
        test_loss = model.evaluate(model, data_loaders.test_loader, criterion)
        losses.append(test_loss)

    plt.plot(losses, '-o')
    plt.xlabel("epoch")
    plt.ylabel("losses")
    plt.title("Losses vs. No. of epochs")
    plt.show()
    torch.save(model.state_dict(), "saved/saved_model.pkl",  _use_new_zipfile_serialization=True)

def main():
    no_epochs = 500
    
    train_model(no_epochs)

if __name__ == '__main__':
    main()
    
