import dgl
import torch
from dgl.data import citation_graph as citegrhs
from dgl import DGLGraph
import torch as th
import pandas as pd
import numpy as np
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))
g.edata["w"] =th.ones(g.num_edges(), dtype=th.int32)

print(g.edata['w'])