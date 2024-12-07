import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from typing import Any, Literal, Optional, Sequence, Tuple, Union

from dltoolbox.transforms import Transformer


class H5DatasetDisk(Dataset):
    """HDF5 dataset class that reads from disk when indexed"""

    def __init__(self,
                 h5_path: str,
                 h5_data_key: str = "data",
                 h5_label_key: str = "labels",
                 select_indices: Optional[Sequence[int]] = None,
                 static_label: Optional[Union[np.ndarray, Tensor]] = None,
                 data_row_dim: int = 0,
                 label_row_dim: int = 0,
                 data_transform: Optional[Transformer] = None,
                 label_transform: Optional[Transformer] = None,
                 ) -> None:
        self.h5_data_key = h5_data_key
        self.h5_label_key = h5_label_key
        self.selected_indices = select_indices
        self.static_label = static_label
        self.data_row_dim = data_row_dim
        self.label_row_dim = label_row_dim
        self.data_transform = data_transform
        self.label_transform = label_transform

        # check for user block in HDF5 file
        self._ub_size: int = 0
        self._ub_bytes: Optional[bytes] = None
        with h5py.File(h5_path, "r") as h5_file:
            self._ub_size = h5_file.userblock_size

        # read user block if available
        if self._ub_size > 0:
            with open(h5_path, "br") as h5_file:
                self._ub_bytes = h5_file.read(self._ub_size)

        self.h5_file = h5py.File(h5_path, "r")
        assert h5_data_key in self.h5_file and type(self.h5_file[h5_data_key]) == h5py.Dataset
        if self.static_label is None:
            assert h5_label_key in self.h5_file and type(self.h5_file[h5_label_key]) == h5py.Dataset
            assert self.h5_file[h5_data_key].shape[data_row_dim] == self.h5_file[h5_label_key].shape[label_row_dim]

        self.h5_data_len: int
        if select_indices is not None:
            self.h5_data_len = len(select_indices)
        else:
            self.h5_data_len = self.h5_file[h5_data_key].shape[data_row_dim]

    def _data(self) -> h5py.Dataset:
        return self.h5_file[self.h5_data_key]

    def _labels(self) -> h5py.Dataset:
        return self.h5_file[self.h5_label_key]

    @staticmethod
    def _h5_take(dataset: h5py.Dataset, index: int, axis: int) -> np.array:
        slices = [index if d == axis else slice(None) for d in range(dataset.ndim)]
        return dataset[tuple(slices)]

    @property
    def ub_size(self) -> int:
        return self._ub_size

    @property
    def ub_bytes(self) -> Optional[bytes]:
        return self._ub_bytes

    def __len__(self) -> int:
        return self.h5_data_len

    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]:
        if self.selected_indices is not None:
            index = self.selected_indices[index]

        data = self._h5_take(self._data(), index, axis=self.data_row_dim)
        label = self.static_label if self.static_label is not None else self._h5_take(self._labels(), index,
                                                                                      axis=self.label_row_dim)

        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return data, label


class H5DatasetMemory(Dataset):
    """HDF5 dataset class that reads the entire dataset into main memory"""

    def __init__(self,
                 h5_path: str,
                 h5_data_key: str = "data",
                 h5_label_key: str = "labels",
                 select_indices: Optional[Sequence[int]] = None,
                 static_label: Optional[Union[np.ndarray, Tensor]] = None,
                 data_row_dim: int = 0,
                 label_row_dim: int = 0,
                 data_transform: Optional[Transformer] = None,
                 label_transform: Optional[Transformer] = None,
                 ) -> None:
        self.data: np.array
        self.labels: np.array
        self.static_label = static_label
        self.data_row_dim: int = data_row_dim
        self.label_row_dim: int = label_row_dim
        self.data_transform = data_transform
        self.label_transform = label_transform

        self._ub_size: int = 0
        self._ub_bytes: Optional[bytes] = None
        # check for user block in HDF5 file
        with h5py.File(h5_path, "r") as h5_file:
            self._ub_size = h5_file.userblock_size

        # read user block if available
        if self._ub_size > 0:
            with open(h5_path, "br") as h5_file:
                self._ub_bytes = h5_file.read(self._ub_size)

        with h5py.File(h5_path, "r") as h5_file:
            assert h5_data_key in h5_file and type(h5_file[h5_data_key]) == h5py.Dataset
            if self.static_label is None:
                assert h5_label_key in h5_file and type(h5_file[h5_label_key]) == h5py.Dataset
                assert h5_file[h5_data_key].shape[data_row_dim] == h5_file[h5_label_key].shape[label_row_dim]

            h5_data: h5py.Dataset = h5_file[h5_data_key]
            if select_indices is not None:
                data_shape = tuple(
                    [s if d != data_row_dim else len(select_indices) for d, s in enumerate(h5_data.shape)]
                )
                data_source_sel = tuple(
                    [slice(None) if d != data_row_dim else select_indices for d, s in enumerate(h5_data.shape)]
                )
                self.data = np.empty(data_shape, dtype=h5_data.dtype)
                h5_data.read_direct(self.data, source_sel=data_source_sel)
            else:
                self.data = np.empty(h5_data.shape, dtype=h5_data.dtype)
                h5_data.read_direct(self.data)

            if self.static_label is None:
                h5_labels: h5py.Dataset = h5_file[h5_label_key]
                if select_indices is not None:
                    labels_shape = tuple(
                        [s if d != label_row_dim else len(select_indices) for d, s in enumerate(h5_labels.shape)]
                    )
                    labels_source_sel = tuple(
                        [slice(None) if d != label_row_dim else select_indices for d, s in enumerate(h5_labels.shape)]
                    )
                    self.labels = np.empty(labels_shape, dtype=h5_labels.dtype)
                    h5_labels.read_direct(self.labels, source_sel=labels_source_sel)
                else:
                    self.labels = np.empty(h5_labels.shape, dtype=h5_labels.dtype)
                    h5_labels.read_direct(self.labels)

                assert self.data.shape[data_row_dim] == self.labels.shape[label_row_dim]

    @property
    def ub_size(self) -> int:
        return self._ub_size

    @property
    def ub_bytes(self) -> Optional[bytes]:
        return self._ub_bytes

    def __len__(self) -> int:
        return self.data.shape[self.data_row_dim]

    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]:
        data = np.take(self.data, index, axis=self.data_row_dim)
        label = self.static_label if self.static_label is not None else np.take(self.labels, index,
                                                                                axis=self.label_row_dim)

        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return data, label


class H5Dataset(Dataset):
    def __init__(self,
                 mode: Literal["disk", "memory"],
                 h5_path: str,
                 h5_data_key: str = "data",
                 h5_label_key: str = "labels",
                 select_indices: Optional[Sequence[int]] = None,
                 static_label: Optional[Union[np.ndarray, Tensor]] = None,
                 data_row_dim: int = 0,
                 label_row_dim: int = 0,
                 data_transform: Optional[Transformer] = None,
                 label_transform: Optional[Transformer] = None,
                 ) -> None:
        if mode == "disk":
            self._instance = H5DatasetDisk(h5_path=h5_path,
                                           h5_data_key=h5_data_key,
                                           h5_label_key=h5_label_key,
                                           select_indices=select_indices,
                                           static_label=static_label,
                                           data_row_dim=data_row_dim,
                                           label_row_dim=label_row_dim,
                                           data_transform=data_transform,
                                           label_transform=label_transform)
        elif mode == "memory":
            self._instance = H5DatasetMemory(h5_path=h5_path,
                                             h5_data_key=h5_data_key,
                                             h5_label_key=h5_label_key,
                                             select_indices=select_indices,
                                             static_label=static_label,
                                             data_row_dim=data_row_dim,
                                             label_row_dim=label_row_dim,
                                             data_transform=data_transform,
                                             label_transform=label_transform)
        else:
            raise ValueError(f'unknown mode "{mode}"')

    @property
    def ub_size(self) -> int:
        return self._instance.ub_size

    @property
    def ub_bytes(self) -> Optional[bytes]:
        return self._instance.ub_bytes

    def __len__(self) -> int:
        return self._instance.__len__()

    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]:
        return self._instance.__getitem__(index)

    def __getattr__(self, name: str) -> Any:
        # delegate attribute access to the instance
        return getattr(self._instance, name)

    def __setattr__(self, key: str, value: Any) -> None:
        if key != "_instance":
            # delegate to the instance
            setattr(self._instance, key, value)
        else:
            self.__dict__[key] = value


def create_hdf5_file(output_path: str,
                     data_arr: np.ndarray,
                     labels_arr: Optional[np.ndarray],
                     data_key: str = "data",
                     labels_key: str = "labels",
                     ub_bytes: Optional[bytes] = None,
                     ) -> None:
    ub_size: int = 0
    if ub_bytes is not None and len(ub_bytes) > 0:
        ub_size = max(512, int(2 ** np.ceil(np.log2(len(ub_bytes)))))

    with h5py.File(output_path, "w", userblock_size=ub_size, libver="latest") as h5_file:
        h5_file.create_dataset(data_key, data=data_arr)
        if labels_arr is not None:
            h5_file.create_dataset(labels_key, data=labels_arr)

    if ub_size > 0:
        with open(output_path, "br+") as h5_file:
            h5_file.write(ub_bytes)
