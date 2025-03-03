import dataclasses
import json
import math
import pathlib
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import pydantic
from datasets import Dataset, load_dataset
from torch import Tensor, nn
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from typing_extensions import TypedDict

from ..utils.tools import subdir_files_count


@dataclasses.dataclass
class DatasetLoader:
    def train(self, batch_size: int) -> Iterable[Tuple[Any, Any]]:
        for Xs, Ys, _Xs_raw, _Ys_raw in self.train_raw(batch_size):
            yield Xs, Ys

    def test(self, batch_size: int) -> Iterable[Tuple[Any, Any]]:
        for Xs, Ys, _Xs_raw, _Ys_raw in self.test_raw(batch_size):
            yield Xs, Ys

    # batch_size -> ...(Xs, Ys, Xs_raw, Ys_raw)
    #     for nlp Xs = Xs_raw := str[], Ys = Ys_raw := int[]
    #     for cv  Xs := Tensor<3,?,?>[] normalized transformed, Ys := int[]
    #             Xs_raw := Tensor<3,w,h>[], Ys_raw := int[]
    train_raw: Callable[[int], Iterable[Tuple[Any, Any, Any, Any]]]
    test_raw: Callable[[int], Iterable[Tuple[Any, Any, Any, Any]]]
    pass


def __spread_arrow_sizes(paths: List[pathlib.Path], tot_size: int) -> List[int]:
    """Spread `tot_size` items over `paths` as evenly as possible, according to
    their sizes."""

    if len(paths) == 1:
        return [tot_size]
    fsz: List[int] = []
    for path in paths:
        ds = Dataset.from_file(path.absolute().as_posix())
        fsz.append(len(ds))
    tot_fsz = sum(fsz)
    ret = [min(math.floor(tot_size * f / tot_fsz), f) for f in fsz]
    while sum(ret) < tot_size and sum(ret) < tot_fsz:
        for i in range(len(ret)):
            if ret[i] < fsz[i]:
                ret[i] += 1
            if sum(ret) >= tot_size:
                break
    return ret


def __prepare_arrow_dataset(
    ds_id: str,
    ds_subtype: Optional[str],
    ds_fpath: pathlib.Path,
    ds_file_count: int,
    train_size: int,
    train_fpaths: List[pathlib.Path],
    train_filter: Callable[[int], bool],
    test_size: int,
    test_seed: int,
    test_fpaths: List[pathlib.Path],
    test_filter: Callable[[int], bool],
    get_xs_ys: Callable[[Any], Tuple[list, list, list, list]],
) -> DatasetLoader:
    if not ds_fpath.exists() or subdir_files_count(ds_fpath) < ds_file_count:
        ds_fpath.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(ds_id, name=ds_subtype)
        dataset.save_to_disk(ds_fpath)  # type: ignore

    # TRICK: we shrink the dataset size to make training more epochs possible
    #        to checkpoint more frequently

    train_sizes = __spread_arrow_sizes(train_fpaths, train_size)
    test_sizes = __spread_arrow_sizes(test_fpaths, test_size)

    def ids_train(tot_size: int, pick: int) -> List[int]:
        items = list(filter(train_filter, range(tot_size)))
        train_seed = random.randint(0, 2**32)
        train_gen = random.Random(train_seed)
        train_gen.shuffle(items)
        # print("train items", tot_size, pick)
        return items[:pick]

    def ids_test(tot_size: int, pick: int) -> List[int]:
        items = list(filter(test_filter, range(tot_size)))
        test_gen = random.Random(test_seed)
        test_gen.shuffle(items)
        # print("test items", tot_size, pick)
        return items[:pick]

    def fn(
        paths: List[pathlib.Path],
        ids: Callable[[int, int], List[int]],
        pick_sizes: List[int],
        batch_size: int,
    ) -> Iterable[Tuple[Any, Any, Any, Any]]:
        for path, pick_size in zip(paths, pick_sizes):
            dataset = Dataset.from_file(path.absolute().as_posix())
            pick_ids = ids(len(dataset), pick_size)
            dataset = dataset.select(pick_ids)

            for _, i in enumerate(dataset.iter(batch_size)):
                xs, ys, xs_raw, ys_raw = get_xs_ys(i)
                if not xs or not ys or len(xs) != len(ys):
                    continue
                if len(xs) != len(xs_raw) or len(ys) != len(ys_raw):
                    continue
                yield xs, ys, xs_raw, ys_raw

    return DatasetLoader(
        train_raw=lambda batch_size: fn(
            train_fpaths, ids_train, train_sizes, batch_size
        ),
        test_raw=lambda batch_size: fn(test_fpaths, ids_test, test_sizes, batch_size),
    )


###############################################################################
#   nlp datasets


def _nlp_xs_ys(
    Xs: Union[Callable[[Any], List[Optional[str]]], str],
    Ys: Union[Callable[[Any], List[Optional[int]]], str],
    filter: Union[Callable[[str, int], bool], int],
) -> Callable[[Any], Tuple[list, list, list, list]]:
    if isinstance(Xs, str):
        Xs_label = Xs
        Xs = lambda i: i[Xs_label]  # noqa: E731
    if isinstance(Ys, str):
        Ys_label = Ys
        Ys = lambda i: i[Ys_label]  # noqa: E731
    if isinstance(filter, int):
        f_max = filter
        filter = lambda _t, label: 0 <= label < f_max  # noqa: E731

    def get_xs_ys(item: Any) -> Tuple[list, list, list, list]:
        text, label = Xs(item), Ys(item)
        # WARNING in make_shapley_mask:
        #     n features must be at least 2 to avoid all 0s
        # so we skip super-short texts
        _filtered_text: List[str] = []
        _filtered_label: List[int] = []
        for _t, _l in zip(text, label):
            if not isinstance(_t, str) or not isinstance(_l, int):
                continue
            if not filter(_t, _l):
                continue
            if len(_t) >= 32:
                _filtered_text.append(_t)
                _filtered_label.append(_l)
        return (
            _filtered_text,
            _filtered_label,
            list(_filtered_text),  # raw ver. = copy of actually processed text
            list(_filtered_label),  # actually we don't use them
        )

    return get_xs_ys


def __prepare_nlp_json_dataset(dir: str) -> DatasetLoader:
    this_path = pathlib.Path(__file__).parent
    with open(this_path / dir / "test.json", "r", encoding="utf-8") as f:
        test_samples = json.loads(f.read())

    def loader_fn(batch_size: int) -> Iterable[Tuple[Any, Any, Any, Any]]:
        for i in range(0, len(test_samples), batch_size):
            batch = test_samples[i : i + batch_size]
            inputs = [x["inputs"] for x in batch]
            targets = [x["targets"] for x in batch]
            yield inputs, targets, list(inputs), list(targets)
        return

    return DatasetLoader(train_raw=loader_fn, test_raw=loader_fn)


def load_nlp_samples() -> DatasetLoader:
    return __prepare_nlp_json_dataset("nlp_samples")


def load_yelp_polarity(
    train_size: int, test_size: int, test_seed: int
) -> DatasetLoader:
    path = pathlib.Path(__file__).parent / "./yelp_polarity/"
    return __prepare_arrow_dataset(
        ds_id="fancyzhx/yelp_polarity",
        ds_subtype=None,
        ds_fpath=path,
        ds_file_count=10,
        train_size=train_size,
        train_fpaths=[path / "train" / "data-00000-of-00001.arrow"],
        train_filter=lambda _: True,
        test_size=test_size,
        test_seed=test_seed,
        test_filter=lambda _: True,
        test_fpaths=[path / "test" / "data-00000-of-00001.arrow"],
        get_xs_ys=_nlp_xs_ys(Xs="text", Ys="label", filter=2),
    )


def load_yelp_polarity_mini() -> DatasetLoader:
    return __prepare_nlp_json_dataset("yelp_polarity_mini")


###############################################################################
#   cv datasets


class CvTransformResize(TypedDict):
    height: int
    width: int


class CvTransformRandomCrop(TypedDict):
    height: int
    width: int
    scale: Tuple[float, float]
    p: float


class CvTransformCenterCrop(TypedDict):
    height: int
    width: int


class CvTransformHorizontalFlip(TypedDict):
    p: float


class CvTransformVerticalFlip(TypedDict):
    p: float


class CvTransformColorJitter(TypedDict):
    brightness: float
    contrast: float
    saturation: float
    hue: float


class CvTransforms(pydantic.BaseModel):
    resize: Optional[CvTransformResize] = None
    random_crop: Optional[CvTransformRandomCrop] = None
    center_crop: Optional[CvTransformCenterCrop] = None
    horizontal_flip: Optional[CvTransformHorizontalFlip] = None
    vertical_flip: Optional[CvTransformVerticalFlip] = None
    color_jitter: Optional[CvTransformColorJitter] = None


def _cv_xs_ys(
    Xs: str,
    Ys: str,
    labels: Dict[Any, int],  # re-order labels to match models
    normalize_mean: Tuple[float, float, float],
    normalize_std: Tuple[float, float, float],
    transforms: CvTransforms,
) -> Callable[[Any], Tuple[list, list, list, list]]:
    tf_to_tensor = ToTensor()
    tf_normalize = Normalize(mean=normalize_mean, std=normalize_std)
    tfs_img: List[nn.Module] = []
    if transforms.resize:
        opt = transforms.resize
        tfs_img.append(Resize(size=(opt["height"], opt["width"])))
    if transforms.random_crop:
        opt = transforms.random_crop
        tfs_img.append(
            RandomResizedCrop(size=(opt["height"], opt["width"]), scale=opt["scale"])
        )
    if transforms.center_crop:
        opt = transforms.center_crop
        tfs_img.append(CenterCrop(size=(opt["height"], opt["width"])))
    if transforms.horizontal_flip:
        opt = transforms.horizontal_flip
        tfs_img.append(RandomHorizontalFlip(p=opt["p"]))
    if transforms.vertical_flip:
        opt = transforms.vertical_flip
        tfs_img.append(RandomVerticalFlip(p=opt["p"]))
    if transforms.color_jitter:
        opt = transforms.color_jitter
        tf = ColorJitter(
            brightness=opt["brightness"],
            contrast=opt["contrast"],
            saturation=opt["saturation"],
            hue=opt["hue"],
        )
        tfs_img.append(tf)

    def get_xs_ys(item: Any) -> Tuple[list, list, list, list]:
        _images: List[PIL.Image.Image] = item[Xs]
        _labels: list = item[Ys]
        _proc_images: List[Tensor] = []
        _proc_labels: List[int] = []
        _proc_raw_images: List[Tensor] = []
        for image, _label in zip(_images, _labels):
            if not isinstance(image, (PIL.Image.Image, np.ndarray, Tensor)):
                continue
            label = labels.get(_label, None)
            if label is None:
                continue
            if not isinstance(image, Tensor):
                image = tf_to_tensor(image)
            if image.shape[0] == 1:  # grayscale
                image = image.repeat(3, 1, 1)
            raw_image = image.clone()
            image = tf_normalize(image)
            # TODO: these ought to be moved to models' input processing and not in datasets
            for tf in tfs_img:
                image = tf(image)
            _proc_images.append(image)
            _proc_labels.append(label)
            _proc_raw_images.append(raw_image)
        return _proc_images, _proc_labels, _proc_raw_images, list(_proc_labels)

    return get_xs_ys


def load_imagenette(
    train_size: int, test_size: int, test_seed: int, transforms: CvTransforms
) -> DatasetLoader:
    path = pathlib.Path(__file__).parent / "./imagenette/"
    imagenette_labels = [  # these are from `frgfm/imagenette` dataset
        ("tench", "n01440764"),
        ("English springer", "n02102040"),
        ("cassette player", "n02979186"),
        ("chain saw", "n03000684"),
        ("church", "n03028079"),
        ("French horn", "n03394916"),
        ("garbage truck", "n03417042"),
        ("gas pump", "n03425413"),
        ("golf ball", "n03445777"),
        ("parachute", "n03888257"),
    ]
    all_labels = {  # in our code we order images like this
        "n02979186": 0,
        "n03417042": 1,
        "n01440764": 2,
        "n02102040": 3,
        "n03028079": 4,
        "n03888257": 5,
        "n03394916": 6,
        "n03000684": 7,
        "n03445777": 8,
        "n03425413": 9,
    }
    # we reorder labels to match our code
    all_labels = {
        i: all_labels[iid] for i, (_name, iid) in enumerate(imagenette_labels)
    }

    return __prepare_arrow_dataset(
        ds_id="frgfm/imagenette",
        ds_subtype="full_size",  # '160px', '320px', 'full_size'
        ds_fpath=path,
        ds_file_count=12,
        train_size=train_size,
        train_fpaths=[
            path / "train" / "data-00000-of-00003.arrow",
            path / "train" / "data-00001-of-00003.arrow",
            path / "train" / "data-00002-of-00003.arrow",
        ],
        train_filter=lambda _: True,
        test_size=test_size,
        test_seed=test_seed,
        test_filter=lambda _: True,
        test_fpaths=[path / "validation" / "data-00000-of-00001.arrow"],
        get_xs_ys=_cv_xs_ys(
            Xs="image",
            Ys="label",
            labels=all_labels,
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225),
            transforms=transforms,
        ),
    )


def preload_all_datasets():
    # nlp
    load_yelp_polarity(560000, 38000, 0x3407)
    # cv
    load_imagenette(9469, 3925, 0x3407, CvTransforms())
    return
