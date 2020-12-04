import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import Processtest
from dgl.nn import GATConv,SAGEConv,GINConv

# 消息函数和聚合函数
gcn_msg = fn.u_add_e('h','w','m')
#gcn_msg = fn.copy_src(src='h', out='m')

gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.layer1 = GCNLayer(96, 48)
        self.layer1 =GCNLayer(500,250)  # 原特征
        #self.layer2 = GATConv(16, 7,num_heads=3)  # embedding
        #self.layer2 = SAGEConv(128,7,'mean')
        #self.layer2 = GCNLayer(48,7)
        self.layer2 = GCNLayer(250,3)
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
net = Net()
print(net)


def load_cora():
    # 加载自己的数据集及其边权重，fetures可以提供预先提取的embedding
    data= Processtest.load_cora()
    features = th.FloatTensor(data.features)
    #labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    return g, features,labels,train_mask, test_mask

def load_citseer():
    # 加载自己的数据集及其边权重，fetures可以提供预先提取的embedding
    data= Processtest.load_citseer()
    #features = th.FloatTensor(data.features)
    #labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    return g, features,labels,train_mask, test_mask

def load_pubmed():
    # 加载自己的数据集及其边权重，fetures可以提供预先提取的embedding
    data= Processtest.load_pubmed()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    return g,features,labels, train_mask, test_mask



def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)



import time
import numpy as np
#g,features,labels,train_mask, test_mask = load_cora() # 加载图，标签等等
#g,features,labels,train_mask, test_mask = load_citseer()
g,features,labels,train_mask, test_mask = load_pubmed() 

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
