import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from gnn.graph_transformer import GraphTransformer
from graph_dataset.graph_dataset import loadGraphDataset, splitGraphDataset
from train_evaluate.distributed import (
    cleanup_distributed,
    rank_zero_print,
    setup_distributed,
    wrap_model_for_distributed,
)
from train_evaluate.train_evaluate import (
    distributed_evaluate,
    distributed_throughput_evaluate,
)


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    graphs_path: Path
    checkpoint_path: Path
    one_hot_mapping: dict
    files_per_class: object
    number_nodes: int
    number_eigenvectors: int
    embedding_size: int
    feedforward_scaling: int
    num_heads: int
    num_layers: int
    dropout: float
    num_classes: int
    discrete_labels: tuple
    binary_from_first_label: bool = False


def _resolve(path):
    return (BASE_DIR / path).resolve()


CICDDOS2019_MAPPING = {
    "Benign": 1,
    "SYN": 2,
    "TFTP": 3,
    "UDP": 4,
    "UDP-Lag": 5,
}

TII_SSRC_23_MULTICLASS_MAPPING = {
    "Benign": 1,
    "Bruteforce": 2,
    "DOS": 3,
    "Information Gathering": 4,
}

TII_SSRC_23_BINARY_MAPPING = {
    "Benign": 1,
    "Bruteforce": 2,
    "DOS": 2,
    "Information Gathering": 2,
}

IOT_23_BINARY_MAPPING = {
    "Benign": 1,
    "Malicious": 2,
}

EDGE_IIOTSET_BINARY_MAPPING = {
    "Benign": 1,
    "Backdoor": 2,
    "DDoS": 2,
    "MITM": 2,
    "OS Fingerprinting": 2,
    "Password Attacks": 2,
    "Port Scanning": 2,
    "Ransomware": 2,
    "SQL Injection": 2,
    "Upload Attack": 2,
    "Vulnerability Scanner": 2,
    "XSS": 2,
}


EXPERIMENTS = {
    "cicddos2019-binary": ExperimentConfig(
        name="cicddos2019-binary",
        graphs_path=_resolve(
            "../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs/Size 20"
        ),
        checkpoint_path=_resolve("../../Checkpoints/cicddos2019-binary-20-40-80.pt"),
        one_hot_mapping=CICDDOS2019_MAPPING,
        files_per_class=50,
        number_nodes=20,
        number_eigenvectors=40,
        embedding_size=80,
        feedforward_scaling=20,
        num_heads=10,
        num_layers=4,
        dropout=0.5,
        num_classes=2,
        discrete_labels=("Benign", "Malicious"),
        binary_from_first_label=True,
    ),
    "cicddos2019-multiclass": ExperimentConfig(
        name="cicddos2019-multiclass",
        graphs_path=_resolve(
            "../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs/Size 20"
        ),
        checkpoint_path=_resolve(
            "../../Checkpoints/cicddos2019-multiclass-20-80-160.pt"
        ),
        one_hot_mapping=CICDDOS2019_MAPPING,
        files_per_class=50,
        number_nodes=20,
        number_eigenvectors=80,
        embedding_size=160,
        feedforward_scaling=40,
        num_heads=20,
        num_layers=4,
        dropout=0.5,
        num_classes=5,
        discrete_labels=tuple(CICDDOS2019_MAPPING.keys()),
    ),
    "tii-ssrc-23-binary": ExperimentConfig(
        name="tii-ssrc-23-binary",
        graphs_path=_resolve(
            "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/Graphs/Size 20"
        ),
        checkpoint_path=_resolve("../../Checkpoints/tii-ssrc-23-binary-20-80-160.pt"),
        one_hot_mapping=TII_SSRC_23_BINARY_MAPPING,
        files_per_class=10,
        number_nodes=20,
        number_eigenvectors=80,
        embedding_size=160,
        feedforward_scaling=40,
        num_heads=20,
        num_layers=4,
        dropout=0.5,
        num_classes=2,
        discrete_labels=("Benign", "Malicious"),
    ),
    "tii-ssrc-23-multiclass": ExperimentConfig(
        name="tii-ssrc-23-multiclass",
        graphs_path=_resolve(
            "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/Graphs/Size 20"
        ),
        checkpoint_path=_resolve(
            "../../Checkpoints/tii-ssrc-23-multiclass-20-80-160.pt"
        ),
        one_hot_mapping=TII_SSRC_23_MULTICLASS_MAPPING,
        files_per_class=10,
        number_nodes=20,
        number_eigenvectors=80,
        embedding_size=160,
        feedforward_scaling=40,
        num_heads=20,
        num_layers=4,
        dropout=0.5,
        num_classes=4,
        discrete_labels=tuple(TII_SSRC_23_MULTICLASS_MAPPING.keys()),
    ),
    "iot-23-binary": ExperimentConfig(
        name="iot-23-binary",
        graphs_path=_resolve("../../Datasets/IOT-23/My Preprocessing/Graphs/Size 20"),
        checkpoint_path=_resolve("../../Checkpoints/iot-23-binary-20-40-80.pt"),
        one_hot_mapping=IOT_23_BINARY_MAPPING,
        files_per_class=50,
        number_nodes=20,
        number_eigenvectors=40,
        embedding_size=80,
        feedforward_scaling=20,
        num_heads=10,
        num_layers=4,
        dropout=0.5,
        num_classes=2,
        discrete_labels=tuple(IOT_23_BINARY_MAPPING.keys()),
    ),
    "edge-iiotset-binary": ExperimentConfig(
        name="edge-iiotset-binary",
        graphs_path=_resolve(
            "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/"
            "My Preprocessing/Graphs/Size 20"
        ),
        checkpoint_path=_resolve("../../Checkpoints/edge-iiotset-binary-20-40-80.pt"),
        one_hot_mapping=EDGE_IIOTSET_BINARY_MAPPING,
        files_per_class="all",
        number_nodes=20,
        number_eigenvectors=40,
        embedding_size=80,
        feedforward_scaling=20,
        num_heads=10,
        num_layers=4,
        dropout=0.5,
        num_classes=2,
        discrete_labels=("Benign", "Malicious"),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a GraphTransformer checkpoint with torch.distributed."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        choices=sorted(EXPERIMENTS.keys()),
        help="Experiment configuration to evaluate.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiment names and exit.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "all"),
        default="test",
        help='Dataset split to evaluate. Use "all" to run train+dev+test once.',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Per-process evaluation batch size.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override the checkpoint path from the experiment config.",
    )
    parser.add_argument(
        "--graphs-path",
        type=Path,
        default=None,
        help="Override the preprocessed DGL graph directory.",
    )
    parser.add_argument(
        "--files-per-class",
        default=None,
        help='Override sampled files per class. Use an integer or "all".',
    )
    parser.add_argument(
        "--no-ddp-wrap",
        action="store_true",
        help="Shard data across ranks without wrapping the model in DDP.",
    )
    parser.add_argument(
        "--shard-strategy",
        choices=("round-robin", "contiguous"),
        default="round-robin",
        help="How to split graphs across distributed ranks.",
    )
    parser.add_argument(
        "--throughput",
        action="store_true",
        help="Run a synchronized throughput benchmark instead of metric evaluation.",
    )
    parser.add_argument(
        "--lap-pe-backend",
        choices=("torch", "dgl"),
        default="torch",
        help="Backend for Laplacian positional encodings.",
    )
    parser.add_argument(
        "--lap-pe-sign-flip",
        choices=("deterministic", "random", "none"),
        default="deterministic",
        help="Sign convention for the torch Laplacian PE backend.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Number of per-rank batches to run before timing throughput.",
    )
    parser.add_argument(
        "--include-transfer-in-throughput",
        action="store_true",
        help="Include CPU-to-GPU graph transfer inside the timed throughput loop.",
    )
    parser.add_argument(
        "--no-save-pickle",
        action="store_true",
        help="Do not write Results/Pickle/<experiment>-results-<split>.pkl.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_experiments:
        for name in sorted(EXPERIMENTS.keys()):
            print(name)
        return
    if args.experiment is None:
        raise SystemExit("Choose an experiment or pass --list-experiments.")

    os.chdir(BASE_DIR)
    context = setup_distributed()

    try:
        config = _apply_overrides(EXPERIMENTS[args.experiment], args)
        _validate_paths(config)
        _create_output_dirs(context)

        split_names = _selected_splits(args.split)
        evaluation_sets = _load_splits(config, split_names)
        if args.split == "all":
            evaluation_sets = {
                "all": _combine_evaluation_sets(evaluation_sets),
            }

        model = GraphTransformer(
            number_nodes=config.number_nodes,
            node_features_size=4,
            number_eigenvectors=config.number_eigenvectors,
            embedding_size=config.embedding_size,
            feedforward_scaling=config.feedforward_scaling,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
            device=context.device,
            lap_pe_backend=args.lap_pe_backend,
            lap_pe_sign_flip=args.lap_pe_sign_flip,
        ).to(context.device)

        if not args.no_ddp_wrap:
            model = wrap_model_for_distributed(model, context)

        for split_name, (graphs_eval, labels_eval) in evaluation_sets.items():
            rank_zero_print(
                context,
                f"Evaluating {config.name} {split_name} split on "
                f"{context.world_size} process(es)",
            )
            results = _run_split(
                model,
                config,
                split_name,
                graphs_eval,
                labels_eval,
                args,
                context,
            )

            if context.is_main and results is not None:
                if not args.no_save_pickle:
                    _save_results(config.name, split_name, args.throughput, results)

    finally:
        cleanup_distributed()


def _apply_overrides(config, args):
    files_per_class = config.files_per_class
    if args.files_per_class is not None:
        files_per_class = _parse_files_per_class(args.files_per_class)

    return ExperimentConfig(
        name=config.name,
        graphs_path=(args.graphs_path or config.graphs_path).resolve(),
        checkpoint_path=(args.checkpoint or config.checkpoint_path).resolve(),
        one_hot_mapping=config.one_hot_mapping,
        files_per_class=files_per_class,
        number_nodes=config.number_nodes,
        number_eigenvectors=config.number_eigenvectors,
        embedding_size=config.embedding_size,
        feedforward_scaling=config.feedforward_scaling,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=config.num_classes,
        discrete_labels=config.discrete_labels,
        binary_from_first_label=config.binary_from_first_label,
    )


def _parse_files_per_class(value):
    if value == "all":
        return value
    return int(value)


def _validate_paths(config):
    if not config.graphs_path.exists():
        raise FileNotFoundError(f"Graph directory does not exist: {config.graphs_path}")
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {config.checkpoint_path}")


def _run_split(
    model,
    config,
    split_name,
    graphs_eval,
    labels_eval,
    args,
    context,
):
    evaluation_mode = {
        "mode": "train-test-dev",
        "set": split_name,
        "name": config.name,
    }
    if args.throughput:
        return distributed_throughput_evaluate(
            model,
            config.checkpoint_path,
            graphs_eval,
            labels_eval,
            args.batch_size,
            context.device,
            evaluation_mode,
            distributed_context=context,
            shard_strategy=args.shard_strategy,
            warmup_batches=args.warmup_batches,
            include_transfer=args.include_transfer_in_throughput,
        )

    return distributed_evaluate(
        model,
        config.checkpoint_path,
        nn.CrossEntropyLoss(),
        graphs_eval,
        labels_eval,
        list(config.discrete_labels),
        args.batch_size,
        context.device,
        evaluation_mode,
        distributed_context=context,
        shard_strategy=args.shard_strategy,
    )


def _save_results(experiment_name, split_name, throughput, results):
    results_path = BASE_DIR / "Results" / "Pickle"
    result_kind = "throughput" if throughput else "results"
    results_path = results_path / f"{experiment_name}-{result_kind}-{split_name}.pkl"
    with open(results_path, "wb") as file:
        pickle.dump(results, file)
    print(f"Saved results to {results_path}")


def _create_output_dirs(context):
    if context.is_main:
        (BASE_DIR / "Results" / "Diagrams").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "Results" / "Pickle").mkdir(parents=True, exist_ok=True)
    if context.enabled:
        torch.distributed.barrier()


def _selected_splits(split):
    if split == "all":
        return ("train", "dev", "test")
    return (split,)


def _combine_evaluation_sets(evaluation_sets):
    combined_graphs = []
    combined_labels = []
    for split_name in ("train", "dev", "test"):
        graphs, labels = evaluation_sets[split_name]
        combined_graphs.extend(graphs)
        combined_labels.append(labels)
    return combined_graphs, torch.cat(combined_labels, dim=0)


def _load_splits(config, split_names):
    graphs, labels = loadGraphDataset(
        _dataset_dir(config.graphs_path),
        config.one_hot_mapping,
        config.files_per_class,
    )
    (
        graphs_train,
        graphs_dev,
        graphs_test,
        labels_train,
        labels_dev,
        labels_test,
    ) = splitGraphDataset(
        graphs,
        labels,
        test_size=0.1,
        dev_size=0.1,
        correct_class_imbalance=False,
        device=torch.device("cpu"),
    )

    available_splits = {
        "train": (graphs_train, labels_train),
        "dev": (graphs_dev, labels_dev),
        "test": (graphs_test, labels_test),
    }
    return {
        split_name: (
            available_splits[split_name][0],
            _transform_labels(available_splits[split_name][1], config),
        )
        for split_name in split_names
    }


def _dataset_dir(path):
    return str(path) + os.sep


def _transform_labels(labels, config):
    if config.binary_from_first_label:
        return torch.cat((labels[:, :1], 1 - labels[:, :1]), dim=1)
    return labels


if __name__ == "__main__":
    main()
