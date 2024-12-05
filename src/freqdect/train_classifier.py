"""Source code to train deepfake detectors in wavelet and pixel space."""

import argparse
import os
import pathlib
import pickle
from datetime import datetime
from typing import Any, Tuple, Literal, List, Generator

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from models import CNN, Regression, compute_parameter_total, save_model
from dltoolbox.dataset import H5Dataset
from dltoolbox.transforms import ComposeWithMode, Permute, RandomCrop, ToTensor


def val_test_loop(
        data_loader,
        model: torch.nn.Module,
        loss_fun,
        make_binary_labels: bool = False,
        device: torch.device = torch.device("cpu"),
        _description: str = "Validation",
) -> Tuple[float, Any]:
    """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Args:
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
            e.g. a test or validation set in a data split.
        model (torch.nn.Module): The model to evaluate.
        loss_fun: The loss function, which is used to measure the loss of the model on the data set
        make_binary_labels (bool): If flag is set, we only classify binarily, i.e. whether an image is real or fake.
            In this case, the label 0 encodes 'real'. All other labels are cosidered fake data, and are set to 1.

    Returns:
        Tuple[float, Any]: The measured accuracy and loss of the model on the data set.
    """
    with torch.no_grad():
        model.eval()
        val_total = 0
        val_ok = 0.0

        for data, labels in iter(data_loader):
            data: torch.Tensor = data.to(device, non_blocking=True, dtype=torch.float32)
            labels: torch.Tensor = labels.to(device, non_blocking=True, dtype=torch.long)

            if make_binary_labels:
                labels[labels > 0] = 1
            labels.squeeze_()

            out = model(data)
            val_loss = loss_fun(torch.squeeze(out), labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], labels)
            val_ok += torch.sum(ok_mask).item()
            val_total += labels.shape[0]
        val_acc = val_ok / val_total
        print("acc", val_acc, "ok", val_ok, "total", val_total)
    return val_acc, val_loss


def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--features",
        choices=["raw", "packets", "all-packets", "fourier", "all-packets-fourier"],
        default="packets",
        help="the representation type",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="input batch size for testing (default: 512)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="weight decay for optimizer (default: 0)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs (default: 20)"
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=200,
        help="number of training steps after which the model is tested on the validation data set (default: 200)",
    )
    parser.add_argument(
        "--dataset-root-dir",
        type=str,
        required=True,
        help="path to directory containing the dataset files"
    )
    parser.add_argument(
        "--dataset-positive-prefix",
        type=str,
        nargs="+",
        help="name prefix of a positive (fake) dataset"
    )
    parser.add_argument(
        "--dataset-negative-prefix",
        type=str,
        nargs="+",
        help="name prefix to negative (real) dataset"
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="the random seed pytorch works with."
    )

    parser.add_argument(
        "--model",
        choices=["regression", "cnn"],
        default="regression",
        help="The model type chosse regression or CNN. Default: Regression.",
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enables a tensorboard visualization.",
    )

    parser.add_argument(
        "--pbar",
        action="store_true",
        help="enables progress bars",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes started by the test and validation data loaders. The training data_loader "
             "uses three times this argument many workers. Hence, this argument should probably be chosen below 10. "
             "Defaults to 2.",
    )

    parser.add_argument(
        "--class-weights",
        type=float,
        metavar="CLASS_WEIGHT",
        nargs="+",
        default=None,
        help="If specified, training samples are weighted based on their class "
             "in the loss calculation. Expects one weight per class.",
    )

    # one should not specify normalization parameters and request their calculation at the same time
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculates mean and standard deviation used in normalization"
             "from the training data",
    )
    return parser.parse_args()


def create_data_loaders(
        fake_datasets: List[Generator[pathlib.Path, None, None]],
        real_datasets: List[Generator[pathlib.Path, None, None]],
        batch_size: int,
        device: torch.device,
        mode: Literal["disk", "memory"] = "disk"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the data loaders needed for training."""

    train_datasets = []
    valid_datasets = []
    test_datasets = []

    transform = ComposeWithMode([
        ToTensor(device),  # (H, W, 3)
        Permute((2, 0, 1)),  # (3, H, W)
        RandomCrop((128, 128)), # (3, 128, 128)
        Permute((1, 2, 0)), # (128, 128, 3)
    ])

    for gen in fake_datasets:
        for path in gen:
            if 'train' in path.name:
                train_datasets.append(
                    H5Dataset(mode, path.as_posix(),
                              static_label=torch.tensor([1]))
                )
            elif 'valid' in path.name:
                valid_datasets.append(
                    H5Dataset(mode, path.as_posix(),
                              static_label=torch.tensor([1]),
                              data_transform=transform)
                )
            elif 'test' in path.name:
                test_datasets.append(
                    H5Dataset(mode, path.as_posix(),
                              static_label=torch.tensor([1]),
                              data_transform=transform)
                )
            else:
                raise RuntimeError("could not infer train/valid/test from file name")
    for gen in real_datasets:
        for path in gen:
            if 'train' in path.name:
                train_datasets.append(
                    H5Dataset(mode, path.as_posix(),
                              static_label=torch.tensor([0]))
                )
            elif 'valid' in path.name:
                valid_datasets.append(
                    H5Dataset(mode, path.as_posix(),
                              static_label=torch.tensor([0]),
                              data_transform=transform)
                )
            elif 'test' in path.name:
                test_datasets.append(
                    H5Dataset(mode, path.as_posix(),
                              static_label=torch.tensor([0]),
                              data_transform=transform)
                )
            else:
                raise RuntimeError("could not infer train/valid/test from file name")


    # compute mean/std on train set
    # TODO


    # set transform in train sets
    for train_set in train_datasets:
        train_set.data_transform = transform

    train_data_loader = DataLoader(
        ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, num_workers=3,
    )
    valid_data_loader = DataLoader(
        ConcatDataset(valid_datasets), batch_size=batch_size, shuffle=False, num_workers=3
    )
    test_loader = DataLoader(
        ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False, num_workers=3
    )

    return train_data_loader, valid_data_loader, test_loader


def main():
    """Trains a model to classify images.

    All settings such as which model to use, parameters, normalization, data set path,
    seed etc. are specified via cmd line args.
    All training, validation and testing results are printed to stdout.
    After the training is done, the results are stored in a pickle dump in the 'log' folder.
    The state_dict of the trained model is stored there as well.

    Raises:
        ValueError: Raised if mean and std values are incomplete or if the number of
            specified class weights does not match the number of classes.

    # noqa: DAR401
    """
    args = _parse_args()
    print(args)

    if args.class_weights and len(args.class_weights) != args.nclasses:
        raise ValueError(
            f"The number of class_weights ({len(args.class_weights)}) must equal "
            f"the number of classes ({args.nclasses})"
        )

    # fix the seed in the interest of reproducible results.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    make_binary_labels = args.nclasses == 2

    root_path = pathlib.Path(args.dataset_root_dir)

    fake_datasets: List[Generator[pathlib.Path, None, None]] = [root_path.glob(f"{prefix}*.hdf5")
                                                                for prefix in args.dataset_positive_prefix]
    real_datasets: List[Generator[pathlib.Path, None, None]] = [root_path.glob(f"{prefix}*.hdf5")
                                                                for prefix in args.dataset_negative_prefix]

    train_data_loader, valid_data_loader, test_data_loader = create_data_loaders(
        fake_datasets,
        real_datasets,
        args.batch_size,
        device,
    )

    validation_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0

    if args.model == "cnn":
        model = CNN(args.nclasses, args.features).to(device)
    else:
        model = Regression(args.nclasses).to(device)

    print("model parameter count:", compute_parameter_total(model))

    loss_fun = torch.nn.NLLLoss(
        weight=torch.tensor(args.class_weights).to(device) if args.class_weights else None
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for e in tqdm(
            range(args.epochs), desc="Epochs", unit="epochs", disable=not args.pbar
    ):
        # iterate over training data.
        for it, batch in enumerate(
                tqdm(
                    iter(train_data_loader),
                    desc="Training",
                    unit="batches",
                    disable=not args.pbar,
                )
        ):
            model.train()
            optimizer.zero_grad()

            data: torch.Tensor = batch[0].to(device, non_blocking=True, dtype=torch.float32)
            labels: torch.Tensor = batch[1].to(device, non_blocking=True, dtype=torch.long)

            if make_binary_labels:
                labels[labels > 0] = 1
            labels.squeeze_()

            out = model(data)
            loss = loss_fun(torch.squeeze(out), labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], labels)
            acc = torch.sum(ok_mask.type(torch.float32)) / len(labels)

            if it % 10 == 0:
                print(
                    "e",
                    e,
                    "it",
                    it,
                    "total",
                    step_total,
                    "loss",
                    loss.item(),
                    "acc",
                    acc.item(),
                )
            loss.backward()
            optimizer.step()
            step_total += 1
            loss_list.append([step_total, e, loss.item()])
            accuracy_list.append([step_total, e, acc.item()])

            # iterate over val batches.
            if step_total % args.validation_interval == 0:
                val_acc, val_loss = val_test_loop(
                    valid_data_loader,
                    model,
                    loss_fun,
                    device=device,
                    make_binary_labels=make_binary_labels,
                )
                validation_list.append([step_total, e, val_acc])
                if validation_list[-1] == 1.0:
                    print("val acc ideal stopping training.")
                    break

    print(validation_list)

    if not os.path.exists("./log/"):
        os.makedirs("./log/")
    model_file = (
            "./log/"
            + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            + "_"
            + str(args.learning_rate)
            + "_"
            + f"{args.epochs}e"
            + "_"
            + str(args.model)
    )
    save_model(model, model_file + "_" + str(args.seed) + ".pt")
    print(model_file, " saved.")

    # Run over the test set.
    print("Training done, start testing....")

    with torch.no_grad():
        test_acc, test_loss = val_test_loop(
            test_data_loader,
            model,
            loss_fun,
            device=device,
            make_binary_labels=make_binary_labels,
            _description="Testing",
        )
        print("test acc", test_acc)

    _save_stats(
        model_file,
        loss_list,
        accuracy_list,
        validation_list,
        test_acc,
        args,
        len(iter(train_data_loader)),
    )


def _save_stats(
        model_file: str,
        loss_list: list,
        accuracy_list: list,
        validation_list: list,
        test_acc: float,
        args,
        iterations_per_epoch: int,
):
    stats_file = model_file + "_" + str(args.seed) + ".pkl"
    try:
        res = pickle.load(open(stats_file, "rb"))
    except OSError as e:
        res = []
        print(
            e,
            "stats.pickle does not exist, \
              creating a new file.",
        )
    res.append(
        {
            "train_loss": loss_list,
            "train_acc": accuracy_list,
            "val_acc": validation_list,
            "test_acc": test_acc,
            "args": args,
            "iterations_per_epoch": iterations_per_epoch,
        }
    )
    pickle.dump(res, open(stats_file, "wb"))
    print(stats_file, " saved.")


if __name__ == "__main__":
    main()
