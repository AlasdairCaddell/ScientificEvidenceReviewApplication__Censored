import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, c) = self.lstm(x)
        return h[-1]

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SiameseNetwork, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        return h1, h2

# create the Siamese network
input_size = len(word2index)
hidden_size = 300
num_layers = 1
net = SiameseNetwork(input_size, hidden_size, num_layers)

# define the loss function and optimizer
criterion = nn.CosineSimilarity()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# train the network
for epoch in range(num_epochs):
    for i, (sent1, sent2, label) in enumerate(train_data):
        sent1 = torch.tensor(sent1, dtype=torch.long)
        sent2 = torch.tensor(sent2, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)
        h1, h2 = net(sent1, sent2)
        loss = criterion(h1, h2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()