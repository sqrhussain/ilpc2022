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

class RoGCNLayer(nn.Module):
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
        use_bias: bool = True,
        activation: Hint[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        self_loop_dropout: float = 0.2,
        decomposition: Hint[Decomposition] = None,
        decomposition_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the layer.
        :param input_dim: >0
            the input dimension
        :param num_relations:
            the number of relations
        :param output_dim: >0
            the output dimension. If none is given, use the input dimension.
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
        # cf. https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/message_gcns/gcn_basis.py#L22-L24  # noqa: E501
        # there are separate decompositions for forward and backward relations.
        # the self-loop weight is not decomposed.
        
        # self.fwd = decomposition_resolver.make(
        #     query=decomposition,
        #     pos_kwargs=decomposition_kwargs,
        #     input_dim=input_dim,
        #     num_relations=num_relations,
        # )
        # output_dim = self.fwd.output_dim
        # self.bwd = decomposition_resolver.make(
        #     query=decomposition,
        #     pos_kwargs=decomposition_kwargs,
        #     input_dim=input_dim,
        #     num_relations=num_relations,
        # )

        # TODO self.output_dim ?

        self.relation_representation = Embedding(max_id=num_relations,
            embedding_dim=input_dim)

        self.num_relations = num_relations

        self.bias = nn.Parameter(torch.empty(output_dim)) if use_bias else None
        self.dropout = nn.Dropout(p=self_loop_dropout)
        if activation is not None:
            activation = activation_resolver.make(query=activation, pos_kwargs=activation_kwargs)
        self.activation = activation

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # nn.init.xavier_normal_(self.relation_representation)

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

        # # self-loop
        # y = self.dropout(x @ self.w_self_loop)
        # # forward messages
        # y = self.fwd(
        #     x=x,
        #     source=source,
        #     target=target,
        #     edge_type=edge_type,
        #     edge_weights=edge_weights,
        #     accumulator=y,
        # )
        # # backward messages
        # y = self.bwd(
        #     x=x,
        #     source=target,
        #     target=source,
        #     edge_type=edge_type,
        #     edge_weights=edge_weights,
        #     accumulator=y,
        # )
        # num_entities = torch.max(torch.cat((source,target)),dim=-1)[0].item()+1
        AdjM = torch.sparse_coo_tensor(torch.stack([source,edge_type]),torch.ones_like(source),(num_entities,self.num_relations),dtype=torch.float)
        # print(AdjM.type())
        D = torch.diag(1/(torch.sparse.sum(AdjM, dim=1).to_dense()+1))
        # print(D)
        y = torch.sparse.mm(AdjM,self.relation_representation())
        y = D @ y
        # print(y.shape)

        if self.bias is not None:
            y = y + self.bias
        # activation
        if self.activation is not None:
            y = self.activation(y)
        return y


decomposition_resolver: ClassResolver[Decomposition] = ClassResolver.from_subclasses(
    base=Decomposition, default=BasesDecomposition
)
