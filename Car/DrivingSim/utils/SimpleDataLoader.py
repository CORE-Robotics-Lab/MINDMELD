import h5py as h5
from torch.utils import data

class MyDataset(data.Dataset):
    def __init__(self,archive,transform=None):
        self.archive=h5.File(archive,'r')
        self.labels=self.archive['label']
        print(self.labels)
        self.data=self.archive['previous_state']
        self.im=self.archive['image']
        self.transform=transform

    def __getitem__(self, index):
        data=self.data[index]
        im=self.im[index]
        if self.transform is not None:
            im=self.transform(im)
        return im,data,self.labels[index]
    def __len__(self):
        return len(self.labels)
