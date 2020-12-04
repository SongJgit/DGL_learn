import dgl
import numpy as np
import scipy
import torch as th
from dgl.data import citation_graph as citegrh
from torch.utils.data import DataLoader

data = citegrh.load_cora()
G = dgl.DGLGraph(data.graph)
labels = th.tensor(data.labels)
# 找出所有标签为0的节点的索引
label0_nodes = th.nonzero(labels == 0).squeeze()
# 找出指向0类的边，in_edges，返回给定节点的入边,第一个tensor是src，第二个就是类为0的dst
src, _ = G.in_edges(label0_nodes)
# 获取与之相连的src的标签
src_labels = labels[src]
# 获得源节点也是同社区的索引
# 找出所有端点都在0类中的边，也就是边的两个节点都归属于这个社区，
intra_src = th.nonzero(src_labels == 0)
# 计算出占比，分母是所有社区A内节点相连的节点数量，也就是与这个社区相连以及内部的
# 边数，分子是这些节点中也属于A社的数量，也就是内部边的数量
print('Intra-class edges percent: %.4f' % (len(intra_src) / len(src_labels)))

train_set = dgl.data.CoraBinary()
G1, pmpd1, label1 = train_set[1]
nx_G1 = G1.to_networkx()
training_loader = DataLoader(train_set,
                             batch_size=1,
                             collate_fn=train_set.collate_fn,
                             drop_last=True)




class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h










# define the optimizer
optimizer = th.optim.Adam(model.parameters(), lr=1e-2)

# A utility function to convert a scipy.coo_matrix to torch.SparseFloat
def sparse2th(mat):
    value = mat.data
    indices = th.LongTensor([mat.row, mat.col])
    tensor = th.sparse.FloatTensor(indices, th.from_numpy(value).float(), mat.shape)
    return tensor

# Train for 20 epochs
for i in range(20):
    all_loss = []
    all_acc = []
    for [g, pmpd, label] in training_loader:
        # Generate the line graph.
        #lg = g.line_graph(backtracking=False)
        # Create torch tensors
        #pmpd = sparse2th(pmpd)
        label = th.from_numpy(label)
        
        # Forward
        
        z = model(g,features)

        # Calculate loss:
        # Since there are only two communities, there are only two permutations
        #  of the community labels.
        loss_perm1 = F.cross_entropy(z, label)
        loss_perm2 = F.cross_entropy(z, 1 - label)
        loss = th.min(loss_perm1, loss_perm2)

        # Calculate accuracy:
        _, pred = th.max(z, 1)
        acc_perm1 = (pred == label).float().mean()
        acc_perm2 = (pred == 1 - label).float().mean()
        acc = th.max(acc_perm1, acc_perm2)
        all_loss.append(loss.item())
        all_acc.append(acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    niters = len(all_loss)
    print("Epoch %d | loss %.4f | accuracy %.4f" % (i,
        sum(all_loss) / niters, sum(all_acc) / niters))
