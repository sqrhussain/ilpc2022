# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Iterable, Optional, Tuple, cast

import torch
from torch import nn

from pykeen.models.inductive import InductiveNodePiece
from pykeen.nn.representation import CompGCNLayer
from pykeen.nn.message_passing import RGCNLayer
from pykeen.typing import HeadRepresentation, InductiveMode, RelationRepresentation, TailRepresentation
from pykeen.utils import get_edge_index

__all__ = [
    "InductiveNodePieceGNN",
]

logger = logging.getLogger(__name__)


class InductiveNodePieceGNN(InductiveNodePiece):
    """Inductive NodePiece with a GNN encoder on top.

    Overall, it's a 3-step procedure:

    1. Featurizing nodes via NodePiece
    2. Message passing over the active graph using NodePiece features
    3. Scoring function for a given batch of triples

    As of now, message passing is expected to be over the full graph
    """

    def __init__(
        self,
        # *,
        # gnn_encoder: Optional[Iterable[nn.Module]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param gnn_encoder:
            an interable of message passing layers. Defaults to 2-layer CompGCN with Hadamard composition.
        :param kwargs:
            additional keyword-based parameters passed to `InductiveNodePiece.__init__`.
        """
        super().__init__(**kwargs)

        train_factory, inference_factory, validation_factory, test_factory = (
            kwargs.get("triples_factory"),
            kwargs.get("inference_factory"),
            kwargs.get("validation_factory"),
            kwargs.get("test_factory"),
        )

        dim = self.entity_representations[0].shape[0]
        self.rgcn_encoder = nn.ModuleList([
            RGCNLayer(
                input_dim=dim,
                num_relations=train_factory.num_relations,
                output_dim=dim,
                activation=torch.nn.ReLU,
            )
            for layer in range(2)
        ])

        self.compgcn_encoder = nn.ModuleList([
            CompGCNLayer(
                input_dim=dim,
                output_dim=dim,
                activation=torch.nn.ReLU,
                dropout=0.1,
            )
            for layer in range(0)
        ])

        # Saving edge indices for all the supplied splits
        assert train_factory is not None, "train_factory must be a valid triples factory"
        self.register_buffer(name="training_edge_index", tensor=get_edge_index(triples_factory=train_factory))
        self.register_buffer(name="training_edge_type", tensor=train_factory.mapped_triples[:, 1])

        if inference_factory is not None:
            inference_edge_index = get_edge_index(triples_factory=inference_factory)
            inference_edge_type = inference_factory.mapped_triples[:, 1]

            self.register_buffer(name="validation_edge_index", tensor=inference_edge_index)
            self.register_buffer(name="validation_edge_type", tensor=inference_edge_type)
            self.register_buffer(name="testing_edge_index", tensor=inference_edge_index)
            self.register_buffer(name="testing_edge_type", tensor=inference_edge_type)
        else:
            assert (
                validation_factory is not None and test_factory is not None
            ), "Validation and test factories must be triple factories"
            self.register_buffer(
                name="validation_edge_index", tensor=get_edge_index(triples_factory=validation_factory)
            )
            self.register_buffer(name="validation_edge_type", tensor=validation_factory.mapped_triples[:, 1])
            self.register_buffer(name="testing_edge_index", tensor=get_edge_index(triples_factory=test_factory))
            self.register_buffer(name="testing_edge_type", tensor=test_factory.mapped_triples[:, 1])

    def reset_parameters_(self):
        """Reset the GNN encoder explicitly in addition to other params."""
        super().reset_parameters_()
        if getattr(self, "compgcn_encoder", None) is not None:
            for layer in self.compgcn_encoder:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        if getattr(self, "rgcn_encoder", None) is not None:
            for layer in self.rgcn_encoder:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def _get_representations(
        self,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        mode: InductiveMode = None,
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails, in canonical shape with a GNN encoder."""
        entity_representations = self._entity_representation_from_mode(mode=mode)

        # Extract all entity and relation representations
        x_e, x_r = entity_representations[0](), self.relation_representations[0]()

        edge_index = getattr(self, f"{mode}_edge_index")
        edge_type = getattr(self, f"{mode}_edge_type")
        
        # Perform message passing and get updated states
        for layer in self.rgcn_encoder:
            x_e = layer(
                x=x_e,
                # x_r=x_r,
                # source=h,target=t,edge_type=r,
                source=edge_index[0,:],target=edge_index[1,:],edge_type=edge_type,
            )
        for layer in self.compgcn_encoder:
            x_e, x_r = layer(
                x_e=x_e,
                x_r=x_r,
                edge_index=edge_index,
                edge_type=edge_type,
            )

        # Use updated entity and relation states to extract requested IDs
        # TODO I got lost in all the Representation Modules and shape casting and wrote this ;(

        hh, rr, tt = [
            x_e.index_select(dim=0, index=h) if h is not None else x_e,
            x_r.index_select(dim=0, index=r) if r is not None else x_r,
            x_e.index_select(dim=0, index=t) if t is not None else x_e,
        ]

        # normalization
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hh, rr, tt)),
        )