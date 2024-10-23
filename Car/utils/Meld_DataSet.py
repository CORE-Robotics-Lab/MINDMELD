from torch.utils import data
import torch


class Meld_Dataset(data.Dataset):
    def __init__(self,trajs,ids,difference, gts,labels,states,images,transform=None,train_test='train',device='cuda'):
        self.trajs=trajs
        self.ids=ids
        self.differences=difference
        self.gts=gts
        self.labels=labels
        self.states=states
        self.images=images

        self.train_test=train_test
        self.prev_im=[]
        self.device=device


        self.transform=transform

    def __getitem__(self, index):

        traj=torch.tensor(self.trajs[index]).to(torch.float32).to(self.device)
        if len(traj.shape)<2:
            traj=traj.unsqueeze(1)
        idx = torch.tensor(self.ids[index]).to(torch.long).to(self.device)
        label = torch.tensor(self.labels[index]).to(torch.float32).to(self.device)
        state = torch.tensor(self.states[index]).to(torch.float32).to(self.device)

        if self.train_test=='test':
            diff=[]
            gt=[]
        else:
            diff=torch.tensor(self.differences[index]).to(self.device)
            gt = torch.tensor(self.gts[index]).to(self.device)

        im_name=self.images[index]


        return traj,idx,diff,gt,label,state,im_name

    def __len__(self):
        return len(self.states)
