import dgl
import networkx as nx
import pandas as pd
import numpy as np
import torch as th
from model import Net
import torch.nn.functional as F
import process



def create_graph(path):
    '''生成图'''
    edges_data = pd.read_csv(path)
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    weight = th.tensor(edges_data['W'])
    weight = th.cat((weight,weight))
    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)
    g.edata['w'] = weight
    return g




# 数据处理
path = 'data/cora/coraheader.csv'
g = create_graph(path)
hidden_size = 16
classes = 7
features = th.eye(34)
# 生成标签
labeled_nodes = th.tensor([0,33]) 
lab_mask = th.BoolTensor(34)
labels = th.tensor([0,1])









net= Net(inputs_features,hidden_size,classes)
optimizer = th.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(30):
    logits = net(g, features)
    # 我们保存the logits以便于接下来可视化
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # 我们仅仅为标记过的节点计算loss
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

