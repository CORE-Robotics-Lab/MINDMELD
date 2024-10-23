import torch
from torch import nn

torch.manual_seed(0)


class Model(nn.Module):
    # Model defining LFD agent
    def __init__(self, hidden_size1, hidden_size2, hidden_size3):
        super(Model, self).__init__()
        # input layer to hidden layer
        self.hidden1 = nn.Linear(6154, hidden_size1)
        # hidden layer to output layer
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, hidden_size3)
        self.out_t = nn.Linear(hidden_size3, 1)
        self.out_s = nn.Linear(hidden_size3, 1)
        self.relu = nn.ReLU()

        self.layer1_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2_cnn = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(.2)

    def forward(self, im, s):
        im = self.layer1_cnn(im.float())
        im = self.layer2_cnn(im)
        im = self.drop_out(im)
        out = torch.hstack((im.reshape(im.shape[0], -1), s.squeeze(1).float()))
        out = self.hidden1(out)
        out = self.drop_out(out)
        out = self.hidden2(self.relu(out))
        out = self.output(self.relu(out))
        out = torch.tanh(out)

        outthrottle = self.out_t(self.relu(out))
        outSteer = self.out_s(self.relu(out))
        return outSteer, outthrottle
