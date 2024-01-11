import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Data_Loaders import Data_Loaders
from sklearn.metrics import confusion_matrix

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__() 

        self.input_dim, self.output_dim = 6, 1
        hidden_dims = [256, 512, 256]

        self.input_to_hidden1 = nn.Linear(self.input_dim, hidden_dims[0])
        self.hidden1_to_hidden2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.hidden2_to_hidden3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.hidden3_to_output = nn.Linear(hidden_dims[2], self.output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        
        h1 = self.relu( self.input_to_hidden1(input))
        h2 = self.relu( self.hidden1_to_hidden2(h1))
        h3 = self.relu( self.hidden2_to_hidden3(h2) )
        output = self.hidden3_to_output(h3)
        return output
    
    def evaluate(self, model, test_loader, loss_function):
        model.eval()
        with torch.no_grad():
            losses = []
            for idx, sample in enumerate(test_loader):
                output = model(sample['input'])
                loss = loss_function(output.reshape(-1), sample['label'])
                losses.append(loss)
                
        return sum(losses) / len(losses)  


def main():

    model = Action_Conditioned_FF()

    

if __name__ == '__main__':
    main()
