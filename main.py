"""Example workflow."""
import logging
from pathlib import Path

import click
import more_click
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.losses import NSSALoss
from pykeen.models.inductive import InductiveNodePiece, InductiveNodePieceGNN
from InductiveRGCN import InductiveRGCN
from pykeen.trackers import ConsoleResultTracker, WANDBResultTracker
from pykeen.training import SLCWATrainingLoop
from pykeen.typing import TESTING, TRAINING, VALIDATION
from pykeen.utils import resolve_device, set_random_seed
from torch.optim import Adam

from dataset import InductiveLPDataset, Size

HERE = Path(__file__).parent.resolve()
DATA = HERE.joinpath("data")

# fix the seed for reproducibility
set_random_seed(42)


# for GNN layer reproducibility
# when running on a GPU, make sure to set up an env variable as advised in the doc:
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html

# torch.use_deterministic_algorithms(True)


@click.command()
@click.option(
    "-ds",
    "--dataset",
    type=click.Choice(["small", "large"]),
    default="small",
    show_default=True,
    help="The dataset to use",
)
@click.option(
    "-d",
    "--embedding-dim",
    type=int,
    default=100,
    show_default=True,
    help="The dimension of the entity embeddings",
)
@click.option(
    "-t",
    "--tokens",
    type=int,
    default=5,
    show_default=True,
    help="Number of tokens to use in NodePiece",
)
@click.option(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.0001,
    show_default=True,
    help="The learning rate of the optimizer",
)
@click.option(
    "-m",
    "--margin",
    type=float,
    default=15.0,
    show_default=True,
    help="The margin value to use for the negative sampling self-adversarial loss.",
)
@click.option(
    "-n",
    "--num-negatives",
    type=int,
    default=4,
    show_default=True,
    help="The number of negative samples per positive.",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=256,
    show_default=True,
    help="The batch size to use during training",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=100,
    show_default=True,
    help="The number of training epochs",
)
@click.option(
    "--wandb",
    is_flag=True,
    help="Track results with Weights & Biases (requires `wandb` to be installed).",
)
@click.option("--save", is_flag=True, help=f"Save the model in the {DATA} directory")
@click.option(
    "--gnn", is_flag=True, help="Use the Inductive NodePiece model with GCN layers"
)
@more_click.log_level_option()
def main(
    dataset: Size,
    embedding_dim: int,
    tokens: int,
    learning_rate: float,
    margin: float,
    num_negatives: int,
    batch_size: int,
    epochs: int,
    wandb: bool,
    save: bool,
    gnn: bool,
    log_level: str,
):
    """Train an inductive model with NodePiece representations and an optional GNN encoder."""
    # set appropriate log-level
    logging.basicConfig(level=log_level)

    # dataset loading
    dataset = InductiveLPDataset(size=dataset)
    loss = NSSALoss(margin=margin)

    # we have two baselines: InductiveNodePiece and InductiveNodePieceGNN
    # the GNN version uses a 2-layer CompGCN message passing encoder on the training / inference graphs
    # but feel free to create and attach your own GNN encoder via the gnn_encoder argument
    # and new inductive link prediction models in general

    model = InductiveRGCN(
        embedding_dim=embedding_dim,
        triples_factory=dataset.transductive_training,
        inference_factory=dataset.inductive_inference,
        num_tokens=tokens,
        aggregation="mlp",
        loss=loss,
    ).to(resolve_device())
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Space occupied: {model.num_parameter_bytes} bytes")

    if wandb:
        tracker = WANDBResultTracker(
            project="inductive_ilp",  # put here your project and entity
            entity="pykeen",
            config=click.get_current_context().params,
        )
        tracker.start_run()
    else:
        tracker = ConsoleResultTracker()

    # default training regime is negative sampling (SLCWA)
    # you can also use the 1-N regime with the LCWATrainingLoop
    # the LCWA loop does not need negative sampling kwargs, but accepts label_smoothing in the .train() method
    training_loop = SLCWATrainingLoop(
        triples_factory=dataset.transductive_training,
        model=model,
        optimizer=optimizer,
        result_tracker=tracker,
        negative_sampler_kwargs=dict(
            # affects training speed, the more - the better
            num_negs_per_pos=num_negatives
        ),
        mode=TRAINING,  # must be specified for the inductive setup
    )

    # specifying hits@k values: 1, 3, 5, 10, 100
    valid_evaluator = RankBasedEvaluator(
        mode=VALIDATION,
        metrics=["hits_at_k"]*5,
        metrics_kwargs=[dict(k=k) for k in (1, 3, 5, 10, 100)],
        add_defaults=True,
    )
    test_evaluator = RankBasedEvaluator(
        mode=TESTING,
        metrics=["hits_at_k"] * 5,
        metrics_kwargs=[dict(k=k) for k in (1, 3, 5, 10, 100)],
        add_defaults=True
    )

    # model training and eval on validation starts here
    training_loop.train(
        triples_factory=dataset.transductive_training,
        num_epochs=epochs,
        batch_size=batch_size,
        callbacks="evaluation",
        callback_kwargs=dict(
            evaluator=valid_evaluator,
            evaluation_triples=dataset.inductive_validation.mapped_triples,
            prefix="validation",
            frequency=1,
            additional_filter_triples=dataset.inductive_inference.mapped_triples,
            batch_size=batch_size,
        ),
    )

    # final eval on the test set
    result = test_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.inductive_testing.mapped_triples,
        additional_filter_triples=[
            # filtering of other positive triples
            dataset.inductive_inference.mapped_triples,
            dataset.inductive_validation.mapped_triples,
        ],
        batch_size=batch_size,
    )

    # extracting final metrics
    for metric, metric_label in [
        ("inverse_harmonic_mean_rank", "MRR"),
        *((f"hits_at_{k}", f"Hits@{k}") for k in (100, 10, 5, 3, 1)),
        ("adjusted_arithmetic_mean_rank_index", "AMRI"),
    ]:
        logging.info(f"Test {metric_label:10}: {result.get_metric(name=metric):.5f}")

    # you can also log the final results to wandb if you want
    if wandb:
        tracker.log_metrics(
            metrics=result.to_flat_dict(),
            step=epochs + 1,
            prefix="test",
        )

    # saving the final model
    if save:
        torch.save(model, DATA.joinpath("model.pth"))


if __name__ == "__main__":
    main()
