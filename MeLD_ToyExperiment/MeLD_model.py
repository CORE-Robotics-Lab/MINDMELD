import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_size=14, fc1_size=64,fc2_size=32, out_size=10):
        super().__init__()

        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size, fc1_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(fc1_size,fc2_size)
        self.fc3=nn.Linear(fc2_size,out_size)

    def forward(self, z,w):
        x=torch.cat((z,w),dim=1)
        hidden = self.fc1(x)
        hidden = self.fc2(self.relu(hidden))
        output=self.fc3(self.relu(hidden))

        return output



class Recreate_Model(nn.Module):
    def __init__(self, input_size=14, hidden_size=200, out_size=10):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, out_size)

    def forward(self, z,o_hat):
        x=torch.cat((z,o_hat),dim=1)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        return output



class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100,fc1_size=256,fc2_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size


        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first=True,bidirectional=True)#batch, seq, feature

        self.fc1 = nn.Linear(hidden_layer_size*2*10, fc1_size)
        self.fc2=nn.Linear(fc1_size,fc2_size)
        self.fc3=nn.Linear(fc2_size,output_size)
        self.relu=torch.nn.ReLU()

    def forward(self, input_seq,batch_size):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size).cuda(),
                            torch.zeros(1,batch_size, self.hidden_layer_size).cuda())

        lstm_out, h_out = self.lstm(input_seq)
        lstm_out=lstm_out.view(lstm_out.shape[0],lstm_out.shape[1],2,self.hidden_layer_size)
        bi_dir=torch.cat((lstm_out[:,:,0,:],lstm_out[:,:,1,:]),dim=2)
        bi_dir=torch.flatten(bi_dir,start_dim=1)
        predictions = self.fc1(bi_dir.squeeze())
        predictions=self.fc2(self.relu(predictions))
        z=self.fc3(self.relu(predictions))
        return z


class MapModel(nn.Module):
    def __init__(self, input_size=1, z_dim=32, w_dim=2,num_subj=40):
        super().__init__()
        self.recreate=Recreate_Model(input_size=z_dim+input_size,out_size=w_dim)
        self.lstm_embed=LSTM(input_size=input_size,output_size=z_dim)
        self.decode=Decoder(input_size=z_dim+w_dim,out_size=input_size)
        all_w = torch.zeros((num_subj, 2))
        for i in range(num_subj):
            w = torch.rand(2)
            all_w[i, :] = w
        self.w = torch.nn.Parameter(all_w)

    def forward(self, input_seq,w_ind,batch_size=32):
        w_set = self.w[w_ind.long(), :]

        z=self.lstm_embed(input_seq,batch_size)
        if batch_size==1:
            z=z.unsqueeze(0)
        o_hat=self.decode(z,w_set)
        w_hat=self.recreate(z,o_hat)


        return o_hat,w_hat