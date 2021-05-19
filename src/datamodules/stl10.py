import os
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
from torchvision.datasets import STL10
from src.datamodules import stl10_normalization


class ConcatDataset(Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        result = []
        for dataset in self.datasets:
            cycled_i = i % len(dataset)
            result.append(dataset[cycled_i])

        return tuple(result)

    def __len__(self):
        return max(len(d) for d in self.datasets)


class STL10DataModule(LightningDataModule):

    name = 'stl10'

    def __init__(
            self,
            data_dir: Optional[str] = None,
            unlabeled_val_split: int = 5000,
            train_val_split: int = 500,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            unlabeled_val_split: how many images from the unlabeled training split to use for validation
            train_val_split: how many images from the labeled training split to use for validation
            num_workers: how many workers to use for loading data
            batch_size: the batch size
        """
        super().__init__(*args, **kwargs)

        self.dims = (3, 96, 96)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.unlabeled_val_split = unlabeled_val_split
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.num_unlabeled_samples = 100000 - unlabeled_val_split
        self.num_labeled_samples = 5000 - train_val_split

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """
        Downloads the unlabeled, train and test split
        """
        STL10(self.data_dir, split='unlabeled', download=True, transform=transforms.ToTensor())
        STL10(self.data_dir, split='train', download=True, transform=transforms.ToTensor())
        STL10(self.data_dir, split='test', download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        """
        Loads the 'unlabeled' split minus a portion set aside for validation via `unlabeled_val_split`.
        """
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = STL10(self.data_dir, split='unlabeled', download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(dataset,
                                        [train_length - self.unlabeled_val_split, self.unlabeled_val_split],
                                        generator=torch.Generator().manual_seed(self.seed))
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def train_dataloader_mixed(self):
        """
        Loads a portion of the 'unlabeled' training data and 'train' (labeled) data.
        both portions have a subset removed for validation via `unlabeled_val_split` and `train_val_split`

        Args:

            batch_size: the batch size
            transforms: a sequence of transforms
        """
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms

        unlabeled_dataset = STL10(self.data_dir,
                                  split='unlabeled',
                                  download=False,
                                  transform=transforms)
        unlabeled_length = len(unlabeled_dataset)
        unlabeled_dataset, _ = random_split(unlabeled_dataset,
                                            [unlabeled_length - self.unlabeled_val_split, self.unlabeled_val_split],
                                            generator=torch.Generator().manual_seed(self.seed))

        labeled_dataset = STL10(self.data_dir, split='train', download=False, transform=transforms)
        labeled_length = len(labeled_dataset)
        labeled_dataset, _ = random_split(labeled_dataset,
                                          [labeled_length - self.train_val_split, self.train_val_split],
                                          generator=torch.Generator().manual_seed(self.seed))

        dataset = ConcatDataset(unlabeled_dataset, labeled_dataset)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """
        Loads a portion of the 'unlabeled' training data set aside for validation
        The val dataset = (unlabeled - train_val_split)

        Args:

            batch_size: the batch size
            transforms: a sequence of transforms
        """
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = STL10(self.data_dir, split='unlabeled', download=False, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(dataset,
                                      [train_length - self.unlabeled_val_split, self.unlabeled_val_split],
                                      generator=torch.Generator().manual_seed(self.seed))
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader_mixed(self):
        """
        Loads a portion of the 'unlabeled' training data set aside for validation along with
        the portion of the 'train' dataset to be used for validation

        unlabeled_val = (unlabeled - train_val_split)

        labeled_val = (train- train_val_split)

        full_val = unlabeled_val + labeled_val

        Args:

            batch_size: the batch size
            transforms: a sequence of transforms
        """
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
        unlabeled_dataset = STL10(self.data_dir,
                                  split='unlabeled',
                                  download=False,
                                  transform=transforms)
        unlabeled_length = len(unlabeled_dataset)
        _, unlabeled_dataset = random_split(unlabeled_dataset,
                                            [unlabeled_length - self.unlabeled_val_split, self.unlabeled_val_split],
                                            generator=torch.Generator().manual_seed(self.seed))

        labeled_dataset = STL10(self.data_dir, split='train', download=False, transform=transforms)
        labeled_length = len(labeled_dataset)
        _, labeled_dataset = random_split(labeled_dataset,
                                          [labeled_length - self.train_val_split, self.train_val_split],
                                          generator=torch.Generator().manual_seed(self.seed))

        dataset = ConcatDataset(unlabeled_dataset, labeled_dataset)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        """
        Loads the test split of STL10

        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = STL10(self.data_dir, split='test', download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def train_dataloader_labeled(self):
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = STL10(self.data_dir, split='train', download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(dataset,
                                        [train_length - self.train_val_split, self.train_val_split],
                                        generator=torch.Generator().manual_seed(self.seed))
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader_labeled(self):
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
        dataset = STL10(self.data_dir,
                        split='train',
                        download=False,
                        transform=transforms)
        labeled_length = len(dataset)
        _, labeled_val = random_split(dataset,
                                      [labeled_length - self.train_val_split, self.train_val_split],
                                      generator=torch.Generator().manual_seed(self.seed))

        loader = DataLoader(
            labeled_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            stl10_normalization()
        ])
        return data_transforms
