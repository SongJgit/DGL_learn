import dgl
import numpy as np
import pandas as pd
import torch as th
from dgl.data import DGLDataset
from dgl.data.utils import _get_dgl_url, generate_mask_tensor


class MyDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """
    def __init__(self,
                 name,
                 raw_dir='data/cora',
                 force_reload=False,
                 verbose=False):


        super(MyDataset, self).__init__(name= name,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)


    def download(self):
        # 将原始数据下载到本地磁盘
        pass
    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        # 构建图
        root =self.raw_dir
        #edges_data = pd.read_csv("{}/coraheader.csv".format(root))
        #edges_data = pd.read_csv("{}/citseerheader.csv".format(root))
        edges_data = pd.read_csv("{}/pubmedheader.csv".format(root))
        src = edges_data['SRC'].to_numpy()
        dst = edges_data['DST'].to_numpy()
        g = dgl.graph((src, dst))
        
        # 设置数据集idx
        # cora数据集7类
        """idx_train =np.arange(0,140).tolist()
        idx_test = np.arange(1707,2708).tolist()
        idx_val = np.arange(140,640).tolist()"""

        # citseer6类 3703个特征6类
        """idx_train =np.arange(0,120).tolist()
        idx_test = np.arange(2327,3327).tolist()
        idx_val = np.arange(120,620).tolist()"""
        
        # Pubmed6类 500个特征，3类
        idx_train =np.arange(0,60).tolist()
        idx_test = np.arange(18717,19717).tolist()
        idx_val = np.arange(60,500).tolist()


        # 节点标签
        #labels=th.LongTensor(pd.read_csv("{}/coralabels.csv".format(root),header=None)[1])
        #labels=th.LongTensor(pd.read_csv("{}/citseerlabels.csv".format(root),header=None)[1])
        labels=th.LongTensor(pd.read_csv("{}/pubmedlabels.csv".format(root),header=None)[1])
        g.ndata['labels'] = labels

        # 设置掩码
        train_mask = _sample_mask(idx_train,labels.shape[0])
        val_mask = _sample_mask(idx_val,labels.shape[0])
        test_mask = _sample_mask(idx_test,labels.shape[0])

        #划分掩码
        g.ndata['train_mask'] = generate_mask_tensor(train_mask)
        g.ndata['val_mask'] = generate_mask_tensor(val_mask)
        g.ndata['test_mask'] = generate_mask_tensor(test_mask)



        # 节点特征
        #features = np.load("{}/corafeatures.npy".format(root))
        #features =np.load("{}/citseerfeatures.npy".format(root))
        features =np.load("{}/pubmedfeatures.npy".format(root))
        g.ndata['feat'] = th.FloatTensor(features)

        # 边缘特征
        g.edata['w']= th.tensor(edges_data['W'])
        self._labels = labels
        self._g=g
        self._features = features
        self._train_mask=train_mask
        self._test_mask=test_mask
        self._val_mask = val_mask

    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        #assert idx == 0, "This dataset has only one graph"
        return self._g
        pass

    def __len__(self):
        # 数据样本的数量
        return 1
        pass
    @property
    def features(self):
        #deprecate_property('dataset.feat', 'g.ndata[\'feat\']')
        return self._g.ndata['feat']
    @property
    def graph(self):
        #deprecate_property('dataset.graph', 'dataset.g')
        return self._g

    @property
    def train_mask(self):
        #deprecate_property('dataset.train_mask', 'g.ndata[\'train_mask\']')
        return self._g.ndata['train_mask']

    @property
    def val_mask(self):
        #deprecate_property('dataset.val_mask', 'g.ndata[\'val_mask\']')
        return self._g.ndata['val_mask']

    @property
    def test_mask(self):
        #deprecate_property('dataset.test_mask', 'g.ndata[\'test_mask\']')
        return self._g.ndata['test_mask']

    @property
    def labels(self):
        #deprecate_property('dataset.label', 'g.ndata[\'label\']')
        return self._g.ndata['labels']



def _sample_mask(idx,l):
    """创建mask"""
    mask=np.zeros(l)
    mask[idx]= 1
    return mask


def load_cora(raw_dir=None, force_reload=False, verbose=False):
    name = 'cora'
    raw_dir ='data/after'
    #data = MyDataset(name,raw_dir, force_reload=force_reload, verbose=verbose)
    data = MyDataset(name,raw_dir, force_reload, verbose)
    return data

def load_ppi(raw_dir=None, force_reload=False, verbose=False):
    name = 'ppi'
    raw_dir = "data/ppi"
    data = MyDataset(name,raw_dir, force_reload, verbose)
    return data

def load_citseer(raw_dir=None,force_reload=False,verbose=False):
    name = "citseer"
    raw_dir = 'data/after'
    data = MyDataset(name,raw_dir,force_reload,verbose)
    return data

def load_pubmed(raw_dir=None,force_reload=False,verbose=False):
    name = "pubmed"
    raw_dir = 'data/after'
    data = MyDataset(name,raw_dir,force_reload,verbose)
    return data
