from torch import exp, tensor
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoding_net = nn.Sequential(nn.Linear(400, 5), nn.ReLU())

    def forward(self, x):
        output = self.encoding_net(x)
        return output


class Siamese(nn.Module):
    def __init__(self, embedding_net):
        super(Siamese, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class Discriminator(nn.Module):
    def __init__(self, siamese_net):
        super(Discriminator, self).__init__()
        self.siamese_net = siamese_net

    def forward(self, x1, x2):
        x1, x2 = self.siamese_net(x1, x2)
        z = exp(-(x1 - x2).norm())
        return tensor((z, 1-z))
