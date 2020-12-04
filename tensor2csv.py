import torch as th
import numpy as np
import pandas as pd
from dgl.data import citation_graph as citegrh
from dgl import DGLGraph
# 提取Cora数据集，供后面的骨干度使用
"""out_path = ('data/before')
def load_cora_data():
    data = citegrh.load_cora()
    labels = th.LongTensor(data.labels)
    features = th.FloatTensor(data.features)
    g = DGLGraph(data.graph)
    return g,labels,features
g ,labels,features = load_cora_data()
src = g.edges()[0].numpy()
dst = g.edges()[1].numpy()
src_series = pd.Series(src)
dst_series = pd.Series(dst)
labels = pd.Series(labels.numpy())
np.save("{}/corafeatures.npy".format(out_path),features.numpy())
#labels.to_csv("{}/coralabels.csv".format(out_path),header=None)
#df = pd.DataFrame({'SRC': src_series, 'DST': dst_series})
#df.to_csv("{}/beforecora.csv".format(out_path),header=None,index=None)"""


"""out_path = ('data/before')
def load_citeseer_data():
    data = citegrh.load_citeseer()
    labels = th.LongTensor(data.labels)
    features = th.FloatTensor(data.features)
    g = DGLGraph(data.graph)
    return g,labels,features
g,labels,features= load_citeseer_data()
src = g.edges()[0].numpy()
dst = g.edges()[1].numpy()
src_series = pd.Series(src)
dst_series = pd.Series(dst)
df = pd.DataFrame({'SRC': src_series, 'DST': dst_series})
labels = pd.Series(labels.numpy())
#np.save("{}/citseerfeatures.npy".format(out_path),features.numpy())
#features.to_csv("{}/citseerfeatures.csv".format(out_path),header=None,index=None)
#labels.to_csv("{}/citseerlabels.csv".format(out_path),header=None)
#df.to_csv("{}/beforecitseer.csv".format(out_path),header=None,index=None)
#labels = th.tensor(pd.read_csv("{}/coralabels.csv".format('data/after'),header=None)[1])"""


"""out_path = ('data/before')
def load_Pubmed_data():
    data = citegrh.load_pubmed()
    labels = th.LongTensor(data.labels)
    features = th.FloatTensor(data.features)
    g = DGLGraph(data.graph)
    return g,labels,features
g ,labels,features = load_Pubmed_data()
src = g.edges()[0].numpy()
dst = g.edges()[1].numpy()
src_series = pd.Series(src)
dst_series = pd.Series(dst)
labels = pd.Series(labels.numpy())
np.save("{}/pubmedfeatures.npy".format(out_path),features.numpy()) # 存储features
labels.to_csv("{}/pubmedlabels.csv".format(out_path),header=None) # 存储labels
df = pd.DataFrame({'SRC': src_series, 'DST': dst_series})
df.to_csv("{}/beforepubmed.csv".format(out_path),header=None,index=None) # 存边"""



out_path = ('data/before')
def load_ppi_data():
    data = citegrh.load_pubmed()
    labels = th.LongTensor(data.labels)
    features = th.FloatTensor(data.features)
    g = DGLGraph(data.graph)
    return g,labels,features
g ,labels,features = load_Pubmed_data()
src = g.edges()[0].numpy()
dst = g.edges()[1].numpy()
src_series = pd.Series(src)
dst_series = pd.Series(dst)
labels = pd.Series(labels.numpy())
np.save("{}/pubmedfeatures.npy".format(out_path),features.numpy()) # 存储features
labels.to_csv("{}/pubmedlabels.csv".format(out_path),header=None) # 存储labels
df = pd.DataFrame({'SRC': src_series, 'DST': dst_series})
df.to_csv("{}/beforepubmed.csv".format(out_path),header=None,index=None) # 存边