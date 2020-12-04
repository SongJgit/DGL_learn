import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import NNConv
import Processtest
from dgl.nn import GATConv

def edge_f(efeat):

    return th.nn.Linear(1, 20)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = NNConv(300, 150,edge_f, 'mean')
        self.layer2 = GATConv()
        
        #self.layer3 = th.nn.Linear(1, 200)
    def forward(self, g, features):
        efeat = g.edata['w']
        
        x = F.relu(self.layer1(g,features,efeat))
        #x = self.layer3(g, x)
        return x


net = Net()
print(net)
def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

from dgl.data import citation_graph as citegrh
import networkx as nx
def load_cora():
    data= Processtest.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    return g, features, labels, train_mask, test_mask

import time
import numpy as np
g, features, labels, train_mask, test_mask = load_cora()
g.add_edges(g.nodes(), g.nodes())
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []
for epoch in range(50):
    if epoch >=3:
        t0 = time.time()

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch >=3:
        dur.append(time.time() - t0)
    
    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))
