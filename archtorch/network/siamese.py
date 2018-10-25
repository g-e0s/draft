import torch.tensor
import torch.nn as nn


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoding_net = nn.Sequential(nn.Linear(400, 100), nn.ReLU(),
                                          nn.Linear(100, 5), nn.ReLU())

    def forward(self, x):
        output = self.encoding_net(x.float())
        return output


class Siamese(torch.nn.Module):
    def __init__(self, embedding_net):
        super(Siamese, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)