import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from class_resolver import ClassResolver, Hint, HintOrType, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from torch import nn

from pykeen.nn.init import uniform_norm_p1_
from pykeen.nn.representation import LowRankRepresentation, Representation, Embedding
from pykeen.nn.weighting import EdgeWeighting, edge_weight_resolver
from pykeen.triples import CoreTriplesFactory

from pykeen.nn.message_passing import *

import networkx as nx
import numpy as np

class RaWRGCNLayer(nn.Module):
    r"""
    An RGCN layer from [schlichtkrull2018]_ updated to match the official implementation.
    This layer uses separate decompositions for forward and backward edges (i.e., "normal" and implicitly created
    inverse relations), as well as a separate transformation for self-loops.
    Ignoring dropouts, decomposition and normalization, it can be written as
    .. math ::
        y_i = \sigma(
            W^s x_i
            + \sum_{(e_j, r, e_i) \in \mathcal{T}} W^f_r x_j
            + \sum_{(e_i, r, e_j) \in \mathcal{T}} W^b_r x_j
            + b
        )
    where $b, W^s, W^f_r, W^b_r$ are trainable weights. $W^f_r, W^b_r$ are relation-specific, and commonly enmploy a
    weight-sharing mechanism, cf. Decomposition. $\sigma$ is an activation function. The individual terms in both sums
    are typically weighted. This is implemented by EdgeWeighting. Moreover, RGCN employs an edge-dropout, however,
    this needs to be done outside of an individual layer, since the same edges are dropped across all layers. In
    contrast, the self-loop dropout is layer-specific.
    """

    def __init__(
        self,
        num_relations: int,
        input_dim: int,
        output_dim: Optional[int] = None,
        walk_length: int = 3,
        walk_count: int = 10,
        use_bias: bool = True,
        weighted: bool = False,
        activation: Hint[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        self_loop_dropout: float = 0.2,
        decomposition: Hint[Decomposition] = None,
        decomposition_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the layer.
        :param input_dim: >0
            the input dimension (basically the relation representation size)
            ignored if weighted==False (then we use output_dim as a relation representation size)
        :param num_relations:
            the number of relations
        :param output_dim: >0
            the output dimension. If none is given, use the input dimension. (basically the entity represnetation size)
        :param use_bias:
            whether to use a trainable bias
        :param activation:
            the activation function to use. Defaults to None, i.e., the identity function serves as activation.
        :param activation_kwargs:
            additional keyword-based arguments passed to the activation function for instantiation
        :param self_loop_dropout: 0 <= self_loop_dropout <= 1
            the dropout to use for self-loops
        :param decomposition:
            the decomposition to use, cf. Decomposition and decomposition_resolver
        :param decomposition_kwargs:
            the keyword-based arguments passed to the decomposition for instantiation
        """
        super().__init__()

        if not weighted:
            if output_dim is not None:
                print("WARN: output_dim set back to input_dim*walk_length")
            output_dim = input_dim*walk_length

        self.relation_representation = Embedding(max_id=num_relations*2,
            embedding_dim=input_dim if weighted else output_dim) # TODO: Should this be the output dim? Probably always enforce weighted=True then it should be fine

        self.num_relations = num_relations
        print(f"This should be TWICE as many as it is in the docs 192=={num_relations}??")

        self.walk_length = walk_length
        self.walk_count = walk_count
        print(f"for each node, get {self.walk_count} lists of {self.walk_length} relation ids by random walking")

        self.bias = nn.Parameter(torch.empty(output_dim)) if use_bias else None
        self.dropout = nn.Dropout(p=self_loop_dropout)
        if activation is not None:
            activation = activation_resolver.make(query=activation, pos_kwargs=activation_kwargs)
        self.activation = activation
        self.weighted = weighted
        if self.weighted:
            self.weight_matrix = nn.Parameter(torch.empty((input_dim*walk_length,output_dim)))

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.weighted is not None:
            # nn.init.zeros_(self.weight_matrix)
            nn.init.xavier_normal_(self.weight_matrix)


    def get_random_walks(
        self,
        num_entities,
        source,
        target,
        edge_type,
    ):
        # Build directed graph
        print("Performing random walks")
        G = nx.from_edgelist([(u.item(),v.item(),{"edge_type":r.item()}) for u,v,r in zip(source,target,edge_type)], create_using=nx.DiGraph)
        G.add_edges_from([(v.item(),u.item(),{"edge_type":self.num_relations + r.item()}) for u,v,r in zip(source,target,edge_type)])
        all_walks = []
        mapping = []
        
        for entity in range(num_entities):
            for i in range(self.walk_count):
                u = entity
                walk = []
                for j in range(self.walk_length):
                    # pick one neighbor randomly
                    neighbors = list(G.neighbors(u))
                    # if len(neighbors) == 0:
                    #     print(u)
                    #     break
                    v = np.random.choice(neighbors)
                    walk.append(G.edges[u,v]["edge_type"])
                    u = v
                all_walks.append(walk)
                mapping.append([entity,i*self.walk_length+j])

        mapping = np.array(mapping)
        mapping = torch.sparse_coo_tensor(torch.tensor(mapping.T), torch.ones(len(mapping)), (num_entities,num_entities*self.walk_count), dtype=torch.float)
        return all_walks,mapping




    def forward(
        self,
        # x: torch.FloatTensor,
        num_entities: int,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
    ):
        """
        Calculate enriched entity representations.
        :param x: shape: (num_entities, input_dim)
            The input entity representations.
        :param source: shape: (num_triples,)
            The indices of the source entity per triple.
        :param target: shape: (num_triples,)
            The indices of the target entity per triple.
        :param edge_type: shape: (num_triples,)
            The relation type per triple.
        :param edge_weights: shape: (num_triples,)
            Scalar edge weights per triple.
        :return: shape: (num_entities, output_dim)
            Enriched entity representations.
        """

        # Build random walk. This should be of shape [num_entities*walk_count, walk_length]
        # Also build entity-to-walk mapping within the same function. This should be of size [num_entities, num_entities*walk_count]
        random_walk_ids, entity_walk_mapping = self.get_random_walks(num_entities,source,target,edge_type)

        # Get vector representations of random walks
        # this should be of shape [num_entities*walk_count, walk_length*input_dim]
        random_walk_vecs = torch.tensor([torch.cat([self.relation_representation()[r] for r in walk]) for walk in random_walk_ids]) # TODO should we call forward here?

        # Degree normalization matrix (in this code, this should be equivalent to dividing the final result by walk_count)
        D = torch.diag(1/(torch.sparse.sum(entity_walk_mapping, dim=1).to_dense()+1))

        # Multiplication: D * entity_walk_mapping * random_walk_vecs * weight
        y = torch.sparse.mm(entity_walk_mapping,random_walk_vecs)
        y = D @ y
        if self.weighted:
            y = y @ self.weight_matrix
        if self.bias is not None:
            y = y + self.bias
        # activation
        if self.activation is not None:
            y = self.activation(y)
        return y


decomposition_resolver: ClassResolver[Decomposition] = ClassResolver.from_subclasses(
    base=Decomposition, default=BasesDecomposition
)
