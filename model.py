import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl

gcn_msg = fn.u_mul_e('h','w','m')
gcn_msg = fn.copy_u('h','ms')
gcn_reduce = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer,self).__init__()
        self.linear = nn.Linear(in_feats,out_feats)

    def forward(self,g,feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class Net(nn.Module):
    def __init__(self,in_feats,hidden_size,num_classes):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_size)
        self.layer2 = GCNLayer(hidden_size, num_classes)
    
    def forward(self, g, features):
        h = F.relu(self.layer1(g, features))
        h = self.layer2(g, h)
        return h




