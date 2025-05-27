import pickle

import albumentations as A
import capybara as cb
import numpy as np

DIR = cb.get_curdir(__file__)


class DefaultImageAug:

    def __init__(self, p=0.5):
        self.aug = A.OneOf([
            A.ShiftScaleRotate(),
            A.CoarseDropout(),
            A.ColorJitter(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=p)

    def __call__(self, img: np.ndarray):
        img = self.aug(image=img)['image']
        return img


class CIFAR100DatasetWithLargeImage:

    def __init__(
        self,
        root: str=None,
        mode: str='train',
        image_size: int=32,
        return_tensor: bool=False,
        image_aug_ratio: float=0,
    ):

        if mode not in ['train', 'test']:
            raise ValueError("mode must be either 'train' or 'test'")

        if root is None:
            self.root = DIR / 'cifar-100-python'
        else:
            self.root = root

        self.image_size = image_size
        self.return_tensor = return_tensor
        self.img_aug = DefaultImageAug(p=image_aug_ratio
        )

        # 讀取資料檔案
        with open(f'{self.root}/{mode}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.images = data[b'data']
            self.labels = data[b'fine_labels']
            self.filenames = data[b'filenames']

        self.images = self.images.reshape(-1, 3, 32, 32)  # shape: (N, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = np.transpose(img, (1, 2, 0))
        large_img = cb.imresize(img, (224, 224))
        img = cb.imresize(img, size=self.image_size)

        if self.return_tensor:
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)
            large_img = np.transpose(large_img, (2, 0, 1))
            img = img.astype(np.float32) / 255.  # (3, 32, 32)
            large_img = large_img.astype(np.float32) / 255.  # (3, 224, 224)
            label = np.array(label, dtype=np.int64)
            return img, large_img, label

        return img, large_img, label


class CIFAR100DatasetSimple:

    def __init__(
        self,
        root: str=None,
        mode: str='train',
        image_size: int=32,
        return_tensor: bool=False,
        image_aug_ratio: float=0.5,
    ):

        if mode not in ['train', 'test']:
            raise ValueError("mode must be either 'train' or 'test'")

        if root is None:
            self.root = DIR / 'cifar-100-python'
        else:
            self.root = root

        self.image_size = image_size
        self.return_tensor = return_tensor

        # 讀取資料檔案
        with open(f'{self.root}/{mode}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.images = data[b'data']
            self.labels = data[b'fine_labels']
            self.filenames = data[b'filenames']

        # shape: (N, 3, 32, 32)
        self.images = self.images.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = np.transpose(img, (1, 2, 0)) # (C, H, W) -> (H, W, C)
        img = cb.imresize(img, size=self.image_size)

        if self.return_tensor:
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img = img.astype(np.float32) / 255.  # 直接簡單歸一化到 [0, 1]
            label = np.array(label, dtype=np.int64)
            return img, label

        return img, label


class CIFAR100DatasetMultiScale:

    def __init__(
        self,
        root: str=None,
        mode: str='train',
        image_size: int=32,
        return_tensor: bool=False,
    ):

        if mode not in ['train', 'test']:
            raise ValueError("mode must be either 'train' or 'test'")

        if root is None:
            self.root = DIR / 'cifar-100-python'
        else:
            self.root = root

        self.mode = mode
        self.image_size = image_size
        self.return_tensor = return_tensor

        # 讀取資料檔案
        with open(f'{self.root}/{mode}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.images = data[b'data']
            self.labels = data[b'fine_labels']
            self.filenames = data[b'filenames']

        # shape: (N, 3, 32, 32)
        self.images = self.images.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = np.transpose(img, (1, 2, 0)) # (C, H, W) -> (H, W, C)

        scale_img_1 = np.zeros((32, 32, 3), dtype=np.uint8)
        scale_img_2 = np.zeros((64, 64, 3), dtype=np.uint8)

        if self.mode == 'train':
            if np.random.rand() < 0.5:
                scale_img_1 = cb.imresize(img, size=(32, 32))
                scale = 1
            else:
                scale_img_2 = cb.imresize(img, size=(64, 64))
                scale = 2
        else:  # 'test'
            scale_img_1 = cb.imresize(img, size=(32, 32))
            scale = 1

        if self.return_tensor:
            scale_img_1 = np.transpose(scale_img_1, (2, 0, 1))
            scale_img_2 = np.transpose(scale_img_2, (2, 0, 1))
            scale_img_1 = scale_img_1.astype(np.float32) / 255.
            scale_img_2 = scale_img_2.astype(np.float32) / 255.
            label = np.array(label, dtype=np.int64)
            return scale_img_1, scale_img_2, label, scale

        return scale_img_1, scale_img_2, label, scale


class CIFAR100AugDataset:

    def __init__(
        self,
        root: str=None,
        mode: str='train',
        image_size: int=32,
        return_tensor: bool=False,
        image_aug_ratio: float=0,
    ):

        if mode not in ['train', 'test']:
            raise ValueError("mode must be either 'train' or 'test'")

        if root is None:
            self.root = DIR / 'cifar-100-python'
        else:
            self.root = root

        self.image_size = image_size
        self.return_tensor = return_tensor
        self.img_aug = DefaultImageAug(p=image_aug_ratio)

        # 讀取資料檔案
        with open(f'{self.root}/{mode}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.images = data[b'data']
            self.labels = data[b'fine_labels']
            self.filenames = data[b'filenames']

        self.images = self.images.reshape(-1, 3, 32, 32)  # shape: (N, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = np.transpose(img, (1, 2, 0))

        img = self.img_aug(img=img)
        img = cb.imresize(img, size=self.image_size)

        if self.return_tensor:
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)
            img = img.astype(np.float32) / 255.  # (3, 32, 32)
            label = np.array(label, dtype=np.int64)
            return img, label

        return img, label