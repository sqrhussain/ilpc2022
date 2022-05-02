
from pykeen.models.baseline import EvaluationOnlyModel
from pykeen.triples.utils import get_entities
from torch import nn
import torch
import networkx as nx

class BaselineConstant(EvaluationOnlyModel):

    def __init__(self, triples_factory, inference_factory):
        super().__init__(triples_factory=triples_factory)
        self.dummy = nn.Conv2d(1, 20, 5).to('cpu')
        self.inference_factory = inference_factory

    def score_t(self, hr_batch, *, slice_size=None, mode=None):
        return torch.ones((hr_batch.shape[0],self.inference_factory.num_entities))

    def score_h(self, rt_batch, *, slice_size=None, mode=None):
        return torch.ones((rt_batch.shape[0],self.inference_factory.num_entities))


class BaselineNetworkX(EvaluationOnlyModel):

    def __init__(self, triples_factory, inference_factory, prediction_model=nx.jaccard_coefficient):
        super().__init__(triples_factory=triples_factory)
        self.dummy = nn.Conv2d(1, 20, 5).to('cpu')
        self.inference_factory = inference_factory
        self.mapping = {u:v for u,v in zip(get_entities(inference_factory.triples),
            inference_factory.entities_to_ids(get_entities(inference_factory.triples)))}
        self.G = nx.from_edgelist(inference_factory.triples[:,[0,2]])
        nx.relabel_nodes(self.G, self.mapping, copy=False)
        self.prediction_model = prediction_model


    def score_t(self, hr_batch, *, slice_size=None, mode=None):
        scores = [{int(s[1]):s[2] for s in self.prediction_model(
            self.G,[(u[0].item(),v) for v in sorted(self.G.nodes()) if v!=u[0].item()])}
            for u in hr_batch] # ignoring relation type
        for i in range(hr_batch.shape[0]): scores[i][hr_batch[i,0].item()]=0 # remove self-loops
        scores = torch.tensor([[d[k] for k in sorted(d)] for d in scores])
        return scores


    def score_h(self, rt_batch, *, slice_size=None, mode=None):
        scores = [{int(s[1]):s[2] for s in self.prediction_model(
            self.G,[(u[0].item(),v) for v in sorted(self.G.nodes()) if v!=u[0].item()])}
            for u in rt_batch] # ignoring relation type
        for i in range(rt_batch.shape[0]): scores[i][rt_batch[i,0].item()]=0 # remove self-loops
        scores = torch.tensor([[d[k] for k in sorted(d)] for d in scores])
        return scores
