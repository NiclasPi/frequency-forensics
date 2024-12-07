"""Source code to train deepfake detectors in wavelet and pixel space."""

import argparse
import os
import pathlib
import pickle
from datetime import datetime
from itertools import chain
from typing import Any, Dict, Generator, Literal, Optional, Tuple

import torch
from pywt import Wavelet
from ptwt.packets import WaveletPacket2D, get_freq_order
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from freqdect.models import CNN, Regression, compute_parameter_total, save_model
from freqdect.prepare_dataset import WelfordEstimator
from dltoolbox.dataset import H5Dataset
from dltoolbox.transforms import *


class WPT2d(TransformerBase):
    def __init__(self, wavelet: str, levels: int, mode: str = "reflect", log: bool = True, eps: float = 1e-12):
        self.wavelet = Wavelet(wavelet)
        self.levels = levels
        self.mode = mode
        self.log = log
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        wp_tree = WaveletPacket2D(x, self.wavelet, self.mode, maxlevel=self.levels)
        wp_nodes = list(''.join(str(i) for i in x) for x in chain.from_iterable(get_freq_order(self.levels)))
        wp_list = []

        for node in wp_nodes:
            wp_list.append(wp_tree[node])

        wp_result = torch.stack(wp_list).squeeze()

        if self.log:
            wp_result = torch.log(torch.abs(wp_result) + self.eps)

        return wp_result


class WPT2dAll:
    def __init__(self, wavelet: str, levels: int, mode: str = "reflect", log: bool = True, eps: float = 1e-12):
        self.wavelet = Wavelet(wavelet)
        self.levels = levels
        self.mode = mode
        self.log = log
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        wp_tree = WaveletPacket2D(x, self.wavelet, self.mode, maxlevel=self.levels)

        wp_result = {"raw": x}

        for level in range(1, self.levels + 1):
            wp_nodes = list(''.join(str(i) for i in x) for x in chain.from_iterable(get_freq_order(level)))
            wp_list = []

            for node in wp_nodes:
                wp_list.append(wp_tree[node])

            wp_result[f"packets{level}"] = torch.stack(wp_list).squeeze()
            if self.log:
                wp_result[f"packets{level}"] = torch.log(torch.abs(wp_result[f"packets{level}"]) + self.eps)

        return wp_result


class FFT(TransformerBase):
    def __init__(self, dims: Tuple[int, ...], log: bool = True, eps: float = 1e-12):
        self.dims = dims
        self.log = log
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fftn(x, dim=self.dims)

        if self.log:
            fft = torch.log(torch.abs(fft) + self.eps)
        else:
            fft = torch.cat([torch.real(fft), torch.imag(fft)], dim=-1)

        return fft


def val_test_loop(
        data_loader,
        model: torch.nn.Module,
        loss_fun,
        gpu_transform: Optional[Transformer] = None,
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

            if gpu_transform is not None:
                data = gpu_transform(data)

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

    parser.add_argument(
        "--wavelet-name",
        type=str,
        default="haar",
        help="Specify wavelet for packet features."
    )

    return parser.parse_args()


def create_data_loaders(
        fake_datasets: List[Generator[pathlib.Path, None, None]],
        real_datasets: List[Generator[pathlib.Path, None, None]],
        features: Literal["raw", "packets"],
        batch_size: int,
        num_workers: int,
        device: torch.device,
        wavelet_name: str = "haar",
        wavelet_boundary: str = "boundary",
        mode: Literal["disk", "memory"] = "disk"
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[Transformer]]:
    """Create the data loaders needed for training."""

    train_datasets = []
    valid_datasets = []
    test_datasets = []

    transform = ComposeWithMode([
        Permute((2, 0, 1)),  # (3, H, W)
        RandomCrop((128, 128)),  # (3, 128, 128)
        Permute((1, 2, 0)) if features in ["raw", "fourier"] else NoTransform()  # (128, 128, 3)
    ])

    gpu_transform = None

    if features == "packets":
        print(f"Level-3 wavelet packets wavelet={wavelet_name} and boundary={wavelet_boundary}")
        gpu_transform = Compose([
            WPT2d(wavelet_name, 3, mode=wavelet_boundary),  # (pN=4³=64, B, 3, pH, pW)
            Permute((1, 0, 3, 4, 2)),  # (B, pN, pH, pW, 3)
        ])
    elif features == "all-packets":
        print(f"All wavelet packets of level-3 WPT with wavelet={wavelet_name} and boundary={wavelet_boundary}")
        gpu_transform = Compose([
            WPT2dAll(wavelet_name, 3, mode=wavelet_boundary),  # Dict[str, {raw, (pN, B, 3, pH, pW)}]
            DictTransformApply("raw", Permute((0, 2, 3, 1))),  # (B, 128, 128, 3)
            DictTransformApply("packets1", Permute((1, 0, 3, 4, 2))),  # (B, pN, pH, pW, 3)
            DictTransformApply("packets2", Permute((1, 0, 3, 4, 2))),
            DictTransformApply("packets3", Permute((1, 0, 3, 4, 2))),
        ])
    elif features == "fourier":
        print("Log-scaled FFT coefficients")
        gpu_transform = Compose([
            FFT((-3, -2), log=True),
        ])
    elif features == "all-packets-fourier":
        print(f"All wavelet packets of level-3 WPT with wavelet={wavelet_name} and boundary={wavelet_boundary} "
              f"PLUS real and imaginary FFT coefficients")
        gpu_transform = Compose([
            WPT2dAll(wavelet_name, 3, mode=wavelet_boundary),  # Dict[str, {raw, (pN, B, 3, pH, pW)}]
            DictTransformApply("raw", Permute((0, 2, 3, 1))),  # (B, 128, 128, 3)
            DictTransformClone("raw", "fourier", shallow=True),  # shallow clone pixels to "fourier"
            DictTransformApply("fourier", FFT((-3, -2), log=True)),  # (B, 128, 128, 3)
            DictTransformApply("packets1", Permute((1, 0, 3, 4, 2))),  # (B, pN, pH, pW, 3)
            DictTransformApply("packets2", Permute((1, 0, 3, 4, 2))),
            DictTransformApply("packets3", Permute((1, 0, 3, 4, 2))),
        ])

    for gen in fake_datasets:
        for path in gen:
            if 'train' in path.name:
                train_datasets.append(
                    H5Dataset(mode, path.as_posix(),
                              static_label=torch.tensor([1]),
                              data_transform=transform)
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
                              static_label=torch.tensor([0]),
                              data_transform=transform)
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

    welford = {} if features in ["all-packets", "all-packets-fourier"] else WelfordEstimator()
    transform.set_eval_mode()  # stop randomness
    for batch, _ in DataLoader(ConcatDataset(train_datasets),
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers):
        batch = batch.to(device, non_blocking=True, dtype=torch.float32)
        if gpu_transform is not None:
            batch = gpu_transform(batch)

        if isinstance(batch, dict):
            for k, v in batch.items():
                if k not in welford:
                    welford[k] = WelfordEstimator()
                welford[k].update(v)
        else:
            welford.update(batch)
        if __debug__:
            print("[DEBUG] break out of mean/std computation")
            break

    if not isinstance(welford, dict):
        mean, std = welford.finalize()
        print(f"Computed mean/std on train set:\nmean={mean}\nstd={std}")

        # append normalize transform
        if gpu_transform is not None:
            gpu_transform.append(
                Normalize(mean=mean.to(dtype=torch.float32, device=device),
                          std=std.to(dtype=torch.float32, device=device))
            )
        else:
            transform.extend([
                ToTensor(),  # because mean/std are tensors
                Normalize(mean=mean.to(dtype=torch.float32, device=torch.device("cpu")),
                          std=std.to(dtype=torch.float32, device=torch.device("cpu")))
            ])
    else:
        for k, v in welford.items():
            mean, std = v.finalize()
            print(f"Computed mean/std on train set for '{k}':\nmean={mean}\nstd={std}")

            gpu_transform.append(DictTransformApply(k, Normalize(mean=mean.to(dtype=torch.float32, device=device),
                                                                 std=std.to(dtype=torch.float32, device=device))))

    transform.set_train_mode()  # reset randomness

    # set transform in train sets
    for train_set in train_datasets:
        train_set.data_transform = transform

    train_data_loader = DataLoader(
        ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_data_loader = DataLoader(
        ConcatDataset(valid_datasets), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_data_loader, valid_data_loader, test_loader, gpu_transform


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

    train_data_loader, valid_data_loader, test_data_loader, gpu_transform = create_data_loaders(
        fake_datasets,
        real_datasets,
        args.features,
        args.batch_size,
        args.num_workers,
        device,
        wavelet_name=args.wavelet_name,
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

            if gpu_transform is not None:
                data = gpu_transform(data)

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
                    gpu_transform=gpu_transform,
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
            + "_"
            + str(args.features)
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
            gpu_transform=gpu_transform,
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
