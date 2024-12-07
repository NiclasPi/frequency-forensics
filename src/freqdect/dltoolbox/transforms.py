import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

Transformer = Callable[
    [Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]],
    Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]
]


class TransformerBase(ABC):
    @abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError()


class TransformerWithMode(TransformerBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._eval = False

    @abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError()

    def train_mode(self) -> None:
        self._eval = False

    def eval_mode(self) -> None:
        self._eval = True

    def is_eval(self) -> bool:
        return self._eval


class Compose(TransformerBase):
    """Compose multiple transforms into one callable class."""

    def __init__(self, transforms: List[Transformer]):
        self._transforms = transforms

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        for transform in self._transforms:
            x = transform(x)
        return x

    def __len__(self) -> int:
        return len(self._transforms)

    def __getitem__(self, index: int) -> Transformer:
        return self._transforms[index]

    def append(self, transform: Transformer):
        self._transforms.append(transform)

    def extend(self, transforms: Iterable[Transformer]):
        self._transforms.extend(transforms)

    def insert(self, index: int, transform: Transformer):
        self._transforms.insert(index, transform)


class ComposeWithMode(Compose):
    def set_train_mode(self) -> None:
        for transform in self._transforms:
            if isinstance(transform, TransformerWithMode):
                transform.train_mode()

    def set_eval_mode(self) -> None:
        for transform in self._transforms:
            if isinstance(transform, TransformerWithMode):
                transform.eval_mode()


class NoTransform(TransformerBase):
    """Does nothing."""

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return x


class ToTensor(TransformerBase):
    """Convert an input to a PyTorch tensor (dtype=float32)."""

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=torch.float32, device=self._device)
        else:
            return x.to(dtype=torch.float32, device=self._device)


class DictTransformCreate:
    """Create named outputs from multiple transforms."""

    def __init__(self, transforms: Dict[str, Transformer]):
        self._transforms = transforms

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        return {key: tf(x) for key, tf in self._transforms.items()}


class DictTransformClone:
    """Clone a named input to a new named output. Performs a deep clone by default."""

    def __init__(self, source: str, clone: str, shallow: bool = False):
        self._source = source
        self._clone = clone
        self._shallow = shallow

    def __call__(self, x: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        if self._shallow:
            x[self._clone] = x[self._source]
        else:
            if isinstance(x[self._source], np.ndarray):
                x[self._clone] = np.copy(x[self._source])
            elif isinstance(x[self._source], torch.Tensor):
                x[self._clone] = torch.clone(x[self._source].detach())
            else:
                raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x[self._source])}")
        return x


class DictTransformApply:
    """Apply transform on one named input."""

    def __init__(self, key: str, transform: Transformer):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        x[self._key] = self._transform(x[self._key])
        return x


class Permute(TransformerBase):
    """Permute dimensions of the given input."""

    def __init__(self, dims: Tuple[int, ...]):
        self.dims = dims

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            return np.transpose(x, self.dims)
        elif isinstance(x, torch.Tensor):
            return torch.permute(x, self.dims)
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


class Normalize(TransformerBase):
    """Normalize using the mean and standard deviation. Supports broadcasting to a common shape."""

    def __init__(self, mean: Union[float, np.ndarray, torch.Tensor], std: Union[float, np.ndarray, torch.Tensor]):
        self._mean = mean
        self._std = std

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return (x - self._mean) / self._std


class RandomCrop(TransformerWithMode):
    """Crop the given image at a random location. The input is expected to have shape [..., H, W]."""

    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__()
        self._th, self._tw = size

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        h, w = x.shape[-2:]
        if h < self._th or w < self._tw:
            raise ValueError(f"crop size {(self._th, self._tw)} is larger than input image size {(h, w)}")
        if h == self._th and w == self._tw:
            return x

        if self.is_eval():
            # perform a center crop in eval mode
            i = h - self._th + 1 // 2
            j = w - self._tw + 1 // 2
            return x[..., i:i + self._th, j:j + self._tw]
        else:
            i = np.random.randint(0, h - self._th + 1)
            j = np.random.randint(0, w - self._tw + 1)
            return x[..., i:i + self._th, j:j + self._tw]
