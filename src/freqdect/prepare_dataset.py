"""Speedy preprocessing using batched PyTorch code."""

import argparse
import functools
import os
import pickle
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from typing import Tuple


import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .corruption import (
    blur,
    jpeg_compression,
    noise,
    random_resized_crop,
    random_rotation,
)
from .data_loader import NumpyDataset
from .fourier_math import batch_fourier_preprocessing
from .wavelet_math import batch_packet_preprocessing, identity_processing


Perturbation = namedtuple("Perturbation", ["rotate", "crop", "jpeg", "noise", "blur"])


class WelfordEstimator:
    """Compute running mean and standard deviations.

    The Welford approach greatly reduces memory consumption.
    """

    def __init__(self) -> None:
        """Create a Welfordestimator."""
        self.collapsed_axis: Optional[Tuple[int, ...]] = None

    # estimate running mean and std
    # average all axis except the color channel
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def update(self, batch_vals: torch.Tensor) -> None:
        """Update the running estimation.

        Args:
            batch_vals (torch.Tensor): The current batch element.
        """
        if not self.collapsed_axis:
            self.collapsed_axis = tuple(np.arange(len(batch_vals.shape[:-1])))
            self.count = torch.zeros(1, device=batch_vals.device, dtype=torch.float64)
            self.mean = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float64
            )
            self.std = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float64
            )
            self.m2 = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float64
            )
        self.count += torch.prod(torch.tensor(batch_vals.shape[:-1]))
        delta = torch.sub(batch_vals, self.mean)
        self.mean += torch.sum(delta / self.count, self.collapsed_axis)
        delta2 = torch.sub(batch_vals, self.mean)
        self.m2 += torch.sum(delta * delta2, self.collapsed_axis)

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finish the estimation and return the computed mean and std.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Estimated mean and variance.
        """
        return self.mean, torch.sqrt(self.m2 / self.count)


def get_label_of_folder(
    path_of_folder: Path, binary_classification: bool = False
) -> int:
    """Get the label of the images in a folder based on the folder path.

        We assume:
            A: Orignal data, B: First gan,
            C: Second gan, D: Third gan, E: Fourth gan.
        A working folder structure could look like:
            A_celeba  B_CramerGAN  C_MMDGAN  D_ProGAN  E_SNGAN
        With each folder containing the images from the corresponding
        source.

    Args:
        path_of_folder (Path):  Path string containing only a single
            underscore directly after the label letter.
        binary_classification (bool): If flag is set, we only classify binarily, i.e. whether an image is real or fake.
            In this case, the prefix 'A' indicates real, \
            which is encoded with the label 0. All other folders are considered
            fake data, encoded with the label 1.

    Raises:
       NotImplementedError: Raised if the label letter is unkown.

    Returns:
        int: The label encoded as integer.

    # noqa: DAR401
    """
    label_str = path_of_folder.name.split("_")[0]
    if binary_classification:
        # differentiate original and generated data
        if label_str == "A":
            return 0
        else:
            return 1
    else:
        # the the label based on the path, As are 0s, Bs are 1, etc.
        if label_str == "A":
            label = 0
        elif label_str == "B":
            label = 1
        elif label_str == "C":
            label = 2
        elif label_str == "D":
            label = 3
        elif label_str == "E":
            label = 4
        else:
            raise NotImplementedError(label_str)
        return label


def get_label(path_to_image: Path, binary_classification: bool) -> int:
    """Get the label based on the image path.

       The file structure must be as outlined in the README file.
       We assume:
            A: Orignal data, B: First gan,
            C: Second gan, D: Third gan, E: Fourth gan.
       A working folder structure could look like:
            A_celeba  B_CramerGAN  C_MMDGAN  D_ProGAN  E_SNGAN
       With each folder containing the images from the corresponding source.

    Args:
        path_to_image (Path): Image path string containing only a single
            underscore directly after the label letter.
        binary_classification (bool): If flag is set, we only classify binarily, i.e. whether an image is real or fake.
            In this case, the prefix 'A' indicates real, which is encoded with the label 0.
            All other folders are considered fake data, encoded with the label 1.

    Raises:
        NotImplementedError: Raised if the label letter is unkown.

    Returns:
        int: The label encoded as integer.
    """
    return get_label_of_folder(path_to_image.parent, binary_classification)


def load_perturb_and_stack(
    path_list: np.ndarray,
    perturbation: Perturbation,
    binary_classification: bool = False,
) -> tuple:
    """Transform a lists of paths into a batches of numpy arrays and record their labels.

        If enabled, some images will be corruped here for robustness testing.

    Args:
        path_list (ndarray): An array of Poxis paths strings.
            The stings must follow the convention outlined
            in the get_label function.
        perturbation (Perturbation): Tuple with the image perturbations to apply.
        binary_classification (bool): If flag is set, we only classify binarily,
            i.e. whether an image is real or fake.

    Returns:
        tuple: A numpy array of size
            (preprocessing_batch_size, height, width, 3)
            and a list of length preprocessing_batch_size.
    """
    image_list = []
    label_list = []
    for path_to_image in path_list:
        try:
            image = Image.open(path_to_image)

            if perturbation.rotate:
                image = random_rotation(image)
            if perturbation.crop:
                image = random_resized_crop(image)
            if perturbation.jpeg:
                image = jpeg_compression(image)
            if perturbation.noise:
                image = noise(image)
            if perturbation.blur:
                image = blur(image)

            image_list.append(np.array(image.convert("RGB")))
            label_list.append(np.array(get_label(path_to_image, binary_classification)))
        except UnidentifiedImageError as e:
            print(e, flush=True)
    return np.stack(image_list), label_list


def save_to_disk(
    data_batch: np.ndarray,
    directory: str,
    previous_file_count: int = 0,
    dir_suffix: str = "",
) -> int:
    """Save images to disk using their position on the dataset as filename.

    Args:
        data_batch (np.ndarray): The image batch to store.
        directory (str): The place to store the images at.
        previous_file_count (int, optional): The number of previously stored images.
            Defaults to 0.
        dir_suffix (str): A comment which is attatched to the output directory.

    Returns:
        int: The new total of storage images.
    """
    # loop over the batch dimension
    if not os.path.exists(f"{directory}{dir_suffix}"):
        print("creating", f"{directory}{dir_suffix}", flush=True)
        os.mkdir(f"{directory}{dir_suffix}")
    file_count = previous_file_count
    for pre_processed_image in data_batch:
        with open(f"{directory}{dir_suffix}/{file_count:06}.npy", "wb") as numpy_file:
            np.save(numpy_file, pre_processed_image)
        file_count += 1

    return file_count


def load_process_store(
    file_list,
    preprocessing_batch_size,
    process,
    target_dir,
    label_string,
    perturbation: Perturbation,
    dir_suffix="",
    binary_classification: bool = False,
):
    """Load, process and store a file list according to a processing function.

    Args:
        file_list (list): PosixPath objects leading to source images.
        preprocessing_batch_size (int): The number of files processed at once.
        process (function): The function to use for the preprocessing.
            I.e. a wavelet packet encoding.
        target_dir (string): A directory where to save the processed files.
        label_string (string): A label we add to the target folder.
        perturbation (Perturbation): A tuple with potential image perturbations
            to apply.
    """
    splits = int(len(file_list) / preprocessing_batch_size)
    batched_files = np.array_split(file_list, splits)
    file_count = 0
    directory = str(target_dir) + "_" + label_string
    all_labels = []
    for current_file_batch in batched_files:
        # load, process and store the current batch training set.
        image_batch, labels = load_perturb_and_stack(
            current_file_batch,
            binary_classification=binary_classification,
            perturbation=perturbation,
        )
        all_labels.extend(labels)
        processed_batch = process(image_batch)
        file_count = save_to_disk(processed_batch, directory, file_count, dir_suffix)
        print(file_count, label_string, "files processed", flush=True)

    # save labels
    with open(f"{directory}{dir_suffix}/labels.npy", "wb") as label_file:
        np.save(label_file, np.array(all_labels))


def load_folder(
    folder: Path, train_size: int, val_size: int, test_size: int
) -> np.ndarray:
    """Create posix-path lists for png files in a folder.

    Given a folder containing portable network graphics (*.png) files
    this functions will create Posix-path lists. A train, test, and
    validation set list is created.

    Args:
        folder: Path to a folder with images from the same source, i.e. A_ffhq .
        train_size: Desired size of the training set.
        val_size: Desired size of the validation set.
        test_size: Desired size of the test set.

    Returns:
        Numpy array with the train, validation and test lists, in this order.

    Raises:
        ValueError: if the requested set sizes are not smaller or equal to the number of images available

    # noqa: DAR401
    """
    file_list = list(folder.glob("./*.png")) + list(folder.glob("./*.jpg"))
    if len(file_list) < train_size + val_size + test_size:
        raise ValueError(
            "Requested set sizes must be smaller or equal to the number of images available."
        )

    # split the list into training, validation and test sub-lists.
    train_list = file_list[:train_size]
    validation_list = file_list[train_size : (train_size + val_size)]
    test_list = file_list[(train_size + val_size) : (train_size + val_size + test_size)]
    return np.asarray([train_list, validation_list, test_list], dtype=object)


def pre_process_folder(
    data_folder: str,
    preprocessing_batch_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    perturbataion: Perturbation,
    feature: Optional[str] = None,
    wavelet: str = "db1",
    boundary: str = "reflect",
    missing_label: int = None,
    binary_classification: bool = False,
    gan_split_factor: float = 1.0,
    level: int = 3,
) -> None:
    """Preprocess a folder containing sub-directories with images from different sources.

    All images are expected to have the same size.
    The sub-directories are expected to indicate to label their source in
    their name. For example,  A - for real and B - for GAN generated imagery.

    Args:
        data_folder (str): The folder with the real and gan generated image folders.
        preprocessing_batch_size (int): The batch_size used for image conversion.
        train_size (int): Desired size of the test subset of each folder.
        val_size (int): Desired size of the validation subset of each folder.
        test_size (int): Desired size of the test subset of each folder.
        feature (str): The feature to pre-compute (choose packets, log_packets or raw).
        missing_label (int): label to leave out of training and validation set (choose from {0, 1, 2, 3, 4, None})
        gan_split_factor (float): factor by which the training and validation subset sizes are scaled for each GAN,
            if a missing label is specified.
        jpeg_compression_number (int): jpeg comression factor used for robustness testing.
            Defaults to None.
        rotation_and_crop (bool): If true some images are randomly cropped or rotated.
            Defaults to False.
    """
    data_dir = Path(data_folder)
    if feature == "raw":
        folder_name = f"{data_dir.name}_{feature}"
    else:
        folder_name = f"{data_dir.name}_{feature}_{wavelet}_{boundary}_{level}"
    if perturbataion.jpeg:
        folder_name += "_jpeg"
    if perturbataion.crop:
        folder_name += "_crop"
    if perturbataion.rotate:
        folder_name += "_rotate"
    if perturbataion.noise:
        folder_name += "_noise"
    if perturbataion.blur:
        folder_name += "_blur"

    target_dir = data_dir.parent / folder_name

    if feature == "packets":
        processing_function = functools.partial(
            batch_packet_preprocessing, wavelet=wavelet, mode=boundary, max_lev=level
        )
    elif feature == "log_packets":
        processing_function = functools.partial(
            batch_packet_preprocessing,
            log_scale=True,
            wavelet=wavelet,
            mode=boundary,
            max_lev=level,
        )
    elif feature == "fourier":
        processing_function = functools.partial(batch_fourier_preprocessing)
    elif feature == "log_fourier":
        processing_function = functools.partial(
            batch_fourier_preprocessing, log_scale=True
        )
    else:
        processing_function = identity_processing  # type: ignore

    folder_list = sorted(data_dir.glob("./*"))

    if missing_label is not None:
        # split files in folders into training/validation/test
        func_load_folder = functools.partial(
            load_folder, train_size=train_size, val_size=val_size, test_size=test_size
        )

        train_list = []
        validation_list = []
        test_list = []

        for folder in folder_list:
            if get_label_of_folder(folder) == missing_label:
                test_list.extend(
                    load_folder(folder, train_size=0, val_size=0, test_size=test_size)[
                        2
                    ]
                )
            else:
                # real data
                if get_label_of_folder(folder, binary_classification=True) == 0:
                    train_result, val_result, test_result = load_folder(
                        folder,
                        train_size=train_size,
                        val_size=val_size,
                        test_size=test_size,
                    )
                # generated data
                else:
                    train_result, val_result, test_result = load_folder(
                        folder,
                        train_size=int(train_size * gan_split_factor),
                        val_size=int(val_size * gan_split_factor),
                        test_size=test_size,
                    )
                train_list.extend(train_result)
                validation_list.extend(val_result)
                test_list.extend(test_result)

    else:
        # split files in folders into training/validation/test
        func_load_folder = functools.partial(
            load_folder, train_size=train_size, val_size=val_size, test_size=test_size
        )
        with ThreadPoolExecutor(max_workers=len(folder_list)) as pool:
            result_lst = list(pool.map(func_load_folder, folder_list))
        results = np.asarray(result_lst)

        train_list = [img for folder in results[:, 0] for img in folder]  # type: ignore
        validation_list = [
            img for folder in results[:, 1] for img in folder  # type: ignore
        ]
        test_list = [img for folder in results[:, 2] for img in folder]  # type: ignore

    # fix the seed to make results reproducible.
    random.seed(42)
    random.shuffle(train_list)
    random.shuffle(validation_list)
    random.shuffle(test_list)

    # train_list[0]
    # PosixPath('/nvme/mwolter/celeba/celeba_align_png_cropped/D_ProGAN/ProGAN_00101489.png')

    if missing_label is not None:
        dir_suffix = f"_missing_{missing_label}"
    else:
        dir_suffix = ""

    binary_classification = True if missing_label is not None else binary_classification

    # group the sets into smaller batches to go easy on the memory.
    print("processing validation set.", flush=True)
    load_process_store(
        validation_list,
        preprocessing_batch_size,
        processing_function,
        target_dir,
        "val",
        dir_suffix=dir_suffix,
        binary_classification=binary_classification,
        perturbation=perturbataion,
    )
    print("validation set stored")

    # do not use binary label in test set to make performance measurements on the different classes possible
    print("processing test set", flush=True)
    load_process_store(
        test_list,
        preprocessing_batch_size,
        processing_function,
        target_dir,
        "test",
        dir_suffix=dir_suffix,
        binary_classification=False,
        perturbation=perturbataion,
    )
    print("test set stored")

    print("processing training set", flush=True)
    load_process_store(
        train_list,
        preprocessing_batch_size,
        processing_function,
        target_dir,
        "train",
        dir_suffix=dir_suffix,
        binary_classification=binary_classification,
        perturbation=perturbataion,
    )
    print("training set stored.", flush=True)

    # compute training normalization.
    # load train data and compute mean and std
    print("computing mean and std values.")
    train_data_set = NumpyDataset(f"{target_dir}_train{dir_suffix}")
    welford = WelfordEstimator()
    for img_no in range(train_data_set.__len__()):
        welford.update(train_data_set.__getitem__(img_no)["image"])
    mean, std = welford.finalize()
    print("mean", mean, "std:", std)
    with open(f"{target_dir}_train{dir_suffix}/mean_std.pkl", "wb") as f:
        pickle.dump([mean.numpy(), std.numpy()], f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        type=str,
        help="The folder with the real and gan generated image folders.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=63_000,
        help="Desired size of the training subset of each folder. (default: 63_000).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=5_000,
        help="Desired size of the test subset of each folder. (default: 5_000).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2_000,
        help="Desired size of the validation subset of each folder. (default: 2_000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="The batch_size used for image conversion. (default: 2048).",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--raw",
        "-r",
        help="Save image data as raw image data.",
        action="store_true",
    )
    group.add_argument(
        "--packets",
        "-p",
        help="Save image data as wavelet packets.",
        action="store_true",
    )
    group.add_argument(
        "--log-packets",
        "-lp",
        help="Save image data as log-scaled wavelet packets.",
        action="store_true",
    )
    group.add_argument(
        "--fourier",
        "-f",
        help="Save image data as Fourier coefficients.",
        action="store_true",
    )
    group.add_argument(
        "--log-fourier",
        "-lf",
        help="Save image data as log-scaled Fourier coefficients.",
        action="store_true",
    )

    parser.add_argument(
        "--missing-label",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=None,
        help="leave this label out of the training and validation set. Used to test how the models generalize to new "
        "GANs.",
    )
    parser.add_argument(
        "--binary-classification",
        help="Only create real (0) or fake (1) labels. Every folder prefixed with 'A' will be labelled as real.",
        action="store_true",
    )
    parser.add_argument(
        "--gan-split-factor",
        type=float,
        default=1.0 / 3.0,
        help="scaling factor for GAN subsets in the binary classification split. If a missing label is specified, the"
        " classification task changes to classifying whether the data was generated or not. In this case, the share"
        " of the GAN subsets in the split sets should be reduced to balance both classes (i.e. real and generated"
        " data). So, for each GAN the training and validation split subset sizes are then calculated as the general"
        " subset size in the split (i.e. the size specified by '--train-size' etc.) times this factor."
        " Defaults to 1./3.",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="haar",
        help="The wavelet to use. Choose one from pywt.wavelist(). Defaults to haar.",
    )
    parser.add_argument(
        "--boundary",
        type=str,
        default="reflect",
        help="The boundary treatment method to use. Choose zero, reflect, or boundary. Defaults to reflect.",
    )

    parser.add_argument(
        "--jpeg",
        action="store_true",
        help="Apply jpeg compression to measure the robustness of our method.",
    )

    parser.add_argument(
        "--crop",
        action="store_true",
        help="If set some images will be randomly cropped.",
    )

    parser.add_argument(
        "--rotate",
        action="store_true",
        help="If set some images will be randomly rotated.",
    )

    parser.add_argument(
        "--noise",
        action="store_true",
        help="If set noise will added to some images.",
    )

    parser.add_argument(
        "--blur",
        action="store_true",
        help="If set a gaussian blur will be applied to all images.",
    )

    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="Sets the maximum decomposition level if a packet representation is chosen.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.packets:
        feature = "packets"
    elif args.log_packets:
        feature = "log_packets"
    elif args.fourier:
        feature = "fourier"
    elif args.log_fourier:
        feature = "log_fourier"
    else:
        feature = "raw"

    pre_process_folder(
        data_folder=args.directory,
        preprocessing_batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        feature=feature,
        missing_label=args.missing_label,
        gan_split_factor=args.gan_split_factor,
        wavelet=args.wavelet,
        boundary=args.boundary,
        perturbataion=Perturbation(
            jpeg=args.jpeg,
            crop=args.crop,
            rotate=args.rotate,
            noise=args.noise,
            blur=args.blur,
        ),
        level=args.level,
    )
