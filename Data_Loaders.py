import torch.utils.data as data
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.dataset = np.genfromtxt("saved/training_data.csv", delimiter=",", dtype="float32")
        
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(self.dataset)
        pickle.dump(scaler, open("saved/scaler.pkl", "wb")) 
        #np.savetxt("saved/training_data.csv", normalized_data, delimiter=",") 
        self.x = normalized_data[:,:-1]
        self.y = normalized_data[:,-1]
        lab = LabelEncoder()
        self.y = lab.fit_transform(self.y)
        ros = RandomOverSampler()
        X_ros, y_ros = ros.fit_resample(self.x, self.y)
        self.x = X_ros
        self.y = y_ros
        network_params = np.zeros(shape=(len(self.x),7))
        network_params[:,:-1] = self.x
        network_params[:,-1] = self.y
        np.savetxt("saved/training_data.csv", network_params, delimiter=",")
        
        
        
        

    def __len__(self):
        return self.y.size
        
    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        input = self.x[idx].astype("float32")
        label = self.y[idx].astype("float32")
        sample = {"input": input, "label": label}
        return sample

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        dataset_size = self.nav_dataset.__len__()
        test_split = 0.2
        self.test_size = int(test_split * dataset_size)
        self.train_size = dataset_size - self.test_size
        train_data, test_data = data.random_split(self.nav_dataset,[self.train_size, self.test_size])
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        
    
def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    
    for idx, sample in enumerate(data_loaders.train_loader):
        train_x, train_y = sample['input'], sample['label']
        
        
    for idx, sample in enumerate(data_loaders.test_loader):
        test_x, test_y = sample['input'], sample['label']
    

if __name__ == '__main__':
    main()
