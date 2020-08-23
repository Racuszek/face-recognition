from torch.utils.data import Dataset
import pickle

class EncodingsDataset(Dataset):
    def __init__(self, encodings, transform=None):
        '''
        Args:
            encodings (string) - path to a .pickle file, output of encode_faces.py
            transform (callable, optional) - optional transform to be performed
        '''
        self.samples=[]
        self.encodings_dict = pickle.loads(open(encodings, "rb").read())
        for i in range(len(self.encodings_dict['encodings'])):
        	self.samples.append((self.encodings_dict['encodings'][i], self.encodings_dict['names'][i]))
        # print(type(encodings_dict))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]