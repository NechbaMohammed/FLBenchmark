import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10,CIFAR100, MNIST, FashionMNIST
from torch.utils.data import DataLoader, random_split, Dataset, Subset, ConcatDataset
from torch.optim import SGD, Optimizer
import torch.optim as optim
from typing import Callable, Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig
import numpy as np
from datasets import load_dataset
import math
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from huggingface_hub import login
import matplotlib.pyplot as plt
from collections import OrderedDict,Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
class FCubeDataset(Dataset):
    def __init__(self, hf_dataset):
        """
        Args:
            hf_dataset: A dictionary containing features and labels.
        """
        login(token="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")
        self.features = torch.stack([
            torch.tensor(hf_dataset['feature_1'], dtype=torch.float32),
            torch.tensor(hf_dataset['feature_2'], dtype=torch.float32),
            torch.tensor(hf_dataset['feature_3'], dtype=torch.float32)
        ], dim=1)  # Shape: (num_samples, 3)
        self.targets = torch.tensor(hf_dataset['targets'], dtype=torch.long)
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)
        return self.features[idx], self.targets[idx]

class FEMNISTDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
        ])
        self.targets = [sample['character'] for sample in hf_dataset]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)
            
        sample = self.hf_dataset[idx]
        image = self.transform(sample['image'])
        label = sample['character']
        return image, label
    
class SVHNDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.transform = transforms.ToTensor()
       
        self.targets = torch.tensor(hf_dataset['targets'], dtype=torch.long)
        
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)
        sample = self.hf_dataset[idx]
        # Convert image to tensor
        image = self.transform(sample['image'])
        
        return image, self.targets[idx]
class CINIC10Dataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # Convert grayscale to RGB
        ])
        
        self.targets = torch.tensor(hf_dataset['targets'], dtype=torch.long)
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)
        
        sample = self.hf_dataset[idx]
        
        # Convert image to tensor and ensure 3 channels
        image = self.transform(sample['image'])
        
        return image, self.targets[idx]
class FEDISIC2019Dataset(Dataset):
    def __init__(self, hf_dataset, image_size=(32, 32)):
        self.hf_dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize images to a consistent size
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])
        
        self.targets = torch.tensor(hf_dataset['targets'], dtype=torch.long)
        
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)
        
        sample = self.hf_dataset[idx]
        
        # Convert image to tensor and resize
        image = self.transform(sample['image'])
        
        return image, self.targets[idx]
class AdultCensusDataset(Dataset):
    def __init__(self, hf_dataset, target_column='income', normalize_numerical=True):
        """
        Args:
            hf_dataset: Hugging Face dataset object
            target_column: Name of the target column
            normalize_numerical: Whether to normalize numerical features
        """
        self.hf_dataset = hf_dataset
        self.target_column = target_column
        self.normalize_numerical = normalize_numerical
        
        # Preprocess the dataset
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the dataset by encoding categorical variables and normalizing numerical features"""
        # Convert Hugging Face dataset to pandas DataFrame
        self.data = self.hf_dataset.to_pandas()
        
        # Separate features and target
        self.targets = self.data[self.target_column]
        self.features = self.data.drop(columns=[self.target_column])
        
        # Handle categorical features
        self.categorical_columns = self.features.select_dtypes(include=['object', 'category']).columns
        self.numerical_columns = self.features.select_dtypes(include=['int64', 'float64']).columns
        
        # One-hot encode categorical features
        self.features = pd.get_dummies(self.features, columns=self.categorical_columns, drop_first=True)
        
        # Normalize numerical features
        if self.normalize_numerical:
            self.scaler = StandardScaler()
            self.features[self.numerical_columns] = self.scaler.fit_transform(
                self.features[self.numerical_columns]
            )
        
        # Encode target
        self.targets = self.targets.apply(lambda x: 1 if x == '>50K' else 0)
        
        # Convert features to numpy array of float32
        self.features_array = self.features.astype(np.float32).values
        
        # Convert to tensors
        self.features_tensor = torch.tensor(self.features_array, dtype=torch.float32)
        self.targets_tensor = torch.tensor(self.targets.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)
            
        return self.features_tensor[idx], self.targets_tensor[idx]
    
    def get_feature_dimensions(self):
        """Get the dimensions of the processed features"""
        return self.features_tensor.shape[1]
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        class_counts = np.bincount(self.targets)
        return torch.tensor([1.0 / x for x in class_counts], dtype=torch.float32)
def split_train_test(dataset, test_fraction=0.2, seed=42):
    """
    Split the dataset into train and test sets based on writer_id.
    
    Args:
        dataset: Hugging Face Dataset object (train split).
        test_fraction: Fraction of writers to allocate to the test set.
        seed: Random seed for reproducibility.
    
    Returns:
        train_dataset: Dataset for training.
        test_dataset: Dataset for testing.
    """
    # Group samples by writer_id
    writer_to_samples = defaultdict(list)
    for idx, sample in enumerate(dataset):
        writer_to_samples[sample['writer_id']].append(idx)
    
    # Split writers into train and test
    writer_ids = list(writer_to_samples.keys())
    np.random.seed(seed)
    np.random.shuffle(writer_ids)
    
    num_test_writers = int(len(writer_ids) * test_fraction)
    test_writers = writer_ids[:num_test_writers]
    train_writers = writer_ids[num_test_writers:]
    
    # Create train and test datasets
    train_indices = []
    for writer_id in train_writers:
        train_indices.extend(writer_to_samples[writer_id])
    
    test_indices = []
    for writer_id in test_writers:
        test_indices.extend(writer_to_samples[writer_id])
    
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    
    return train_dataset, test_dataset
def partition_by_writer_id(dataset, num_clients):
    """
    Partition the dataset by writer_id into `num_clients` partitions.
    
    Args:
        dataset: Hugging Face Dataset object.
        num_clients: Number of clients to partition the data into.
    
    Returns:
        List of datasets, one for each client.
    """
    # Group samples by writer_id
    writer_to_samples = defaultdict(list)
    for idx, sample in enumerate(dataset):
        writer_to_samples[sample['writer_id']].append(idx)
    
    # Split writers into `num_clients` groups
    writer_ids = list(writer_to_samples.keys())
    np.random.shuffle(writer_ids)
    writer_groups = np.array_split(writer_ids, num_clients)
    
    # Create client datasets
    client_datasets = []
    for group in writer_groups:
        indices = []
        for writer_id in group:
            indices.extend(writer_to_samples[writer_id])
        client_datasets.append(dataset.select(indices))
    
    return client_datasets

class FederatedDataset:
    def __init__(self, config: DictConfig, num_clients: int, val_ratio: float = 0.1, seed: Optional[int] = 42):
        self.config = config
        self.num_clients = num_clients
        self.val_ratio = val_ratio
        self.seed = seed
        self.trainset, self.testset = self._download_data(config.name)
        self.datasets = self._partition_data()
        
    def print_dataset_stats(self):
        """
        Print statistics about the dataset, including:
        - Number of training instances
        - Number of test instances
        - Number of features
        - Number of classes
        """
        # Number of training and test instances
        if isinstance(self.trainset, list):
            num_train = len(self.trainset[0])
        else:
            num_train = len(self.trainset)
        num_test = len(self.testset)

        # Number of features
        if hasattr(self.trainset, 'get_feature_dimensions'):
            num_features = self.trainset.get_feature_dimensions()
        else:
            # For image datasets, calculate features from the first sample
            
            sample, _ = self.testset[0]
            
            if isinstance(sample, torch.Tensor):
                # Handle both 2D (grayscale) and 3D (RGB) tensors
                if len(sample.shape) == 1:  # Grayscale image (height, width)
                    num_features = sample.shape[0] 
                elif len(sample.shape) == 2:  # Grayscale image (height, width)
                    num_features = sample.shape[0] * sample.shape[1]
                elif len(sample.shape) == 3:  # RGB image (channels, height, width)
                    num_features = sample.shape[0] * sample.shape[1] * sample.shape[2]
                else:
                    raise ValueError(f"Unexpected tensor shape: {sample.shape}")
            else:
                # For non-tensor data (e.g., tabular data)
                num_features = len(sample)
                           

        # Number of classes
        if hasattr(self.testset, 'targets'):
            targets = self.testset.targets
            if isinstance(targets, torch.Tensor):
                num_classes = len(torch.unique(targets))
            elif isinstance(targets, list):
                num_classes = len(set(targets))
            else:
                num_classes = len(np.unique(targets))
        else:
            # For datasets without explicit targets, infer from the first few samples
            unique_labels = set()
            for _, label in self.trainset:
                unique_labels.add(label)
            num_classes = len(unique_labels)

        # Print the statistics
        print(f"Number of training instances: {num_train}")
        print(f"Number of test instances: {num_test}")
        print(f"Number of features: {num_features}")
        print(f"Number of classes: {num_classes}")
         

    def _download_data(self, dataset_name: str) -> Tuple[Dataset, Dataset]:
        """Download the requested dataset."""
        if dataset_name == "cifar10":
            normalize = transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
            
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            x.unsqueeze(0),  # No need to wrap in Variable
                            (4, 4, 4, 4),
                            mode="reflect",
                        ).data.squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.ColorJitter(brightness=0),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            
            # Data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])
            # data prep for test set            
            trainset = CIFAR10(root="data", train=True, download=True, transform=transform_train)
            testset = CIFAR10(root="data", train=False, download=True, transform=transform_test)
        elif dataset_name == "cifar100":
            normalize = transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
            
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            x.unsqueeze(0),  # No need to wrap in Variable
                            (4, 4, 4, 4),
                            mode="reflect",
                        ).data.squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.ColorJitter(brightness=0),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            
            # Data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])
            # data prep for test set            
            trainset = CIFAR100(root="data", train=True, download=True, transform=transform_train)
            testset = CIFAR100(root="data", train=False, download=True, transform=transform_test)
        elif dataset_name == "mnist":
            transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
            transform_train = transforms.Compose([transforms.ToTensor()])
            transform_test = transforms.Compose([transforms.ToTensor()])
            trainset = MNIST(root="data", train=True, download=True, transform=transform)
            testset = MNIST(root="data", train=False, download=True, transform=transform)
        elif dataset_name == "fmnist":
            transform_train = transforms.Compose([transforms.ToTensor()])
            transform_test = transforms.Compose([transforms.ToTensor()])
            trainset = FashionMNIST(root="data", train=True, download=True, transform=transform_train)
            testset = FashionMNIST(root="data", train=False, download=True, transform=transform_test)
        elif dataset_name == "svhn":
            dataset = load_dataset("ufldl-stanford/svhn",  "cropped_digits")
            if 'label' in dataset['train'].column_names:
                dataset = dataset.rename_column('label', 'targets')
            trainset = SVHNDataset(dataset['train'])
            testset = SVHNDataset(dataset['test'])
        elif dataset_name == "fcube":
            dataset = load_dataset("Nechba/fcube")
            if 'target' in dataset['train'].column_names:
                dataset = dataset.rename_column('target', 'targets')
            
            trainset = dataset['train']
            testset = dataset['test']
            
            # Partition the training data into client datasets
            client_datasets = []
            num_samples_per_client = len(trainset) // self.num_clients
            for i in range(self.num_clients):
                client_data = {
                    'feature_1': trainset['feature_1'][i * num_samples_per_client:(i + 1) * num_samples_per_client],
                    'feature_2': trainset['feature_2'][i * num_samples_per_client:(i + 1) * num_samples_per_client],
                    'feature_3': trainset['feature_3'][i * num_samples_per_client:(i + 1) * num_samples_per_client],
                    'targets': trainset['targets'][i * num_samples_per_client:(i + 1) * num_samples_per_client]
                }
                client_datasets.append(FCubeDataset(client_data))
            
            # Wrap the test set
            testset = FCubeDataset({
                'feature_1': testset['feature_1'],
                'feature_2': testset['feature_2'],
                'feature_3': testset['feature_3'],
                'targets': testset['targets']
            })
            
            return client_datasets, testset
            
        elif dataset_name == "femnist":
            dataset = load_dataset("flwrlabs/femnist")
            
            # Split into train and test sets
            train_dataset, test_dataset = split_train_test(dataset['train'], test_fraction=0.2, seed=self.seed)
            
            # Partition train dataset by writer_id
            client_datasets = partition_by_writer_id(train_dataset, self.num_clients)
            
            # Wrap datasets in FEMNISTDataset
            client_datasets = [FEMNISTDataset(ds) for ds in client_datasets]
            testset = FEMNISTDataset(test_dataset)
            
            return client_datasets, testset
        elif dataset_name == "cinic10":
            dataset = load_dataset("flwrlabs/cinic10")
            if 'label' in dataset['train'].column_names:
                dataset = dataset.rename_column('label', 'targets')
            trainset = CINIC10Dataset(dataset['train'])
            testset = CINIC10Dataset(dataset['test'])
        elif dataset_name == "fedisic2019":
            dataset = load_dataset("flwrlabs/fed-isic2019")
            if 'label' in dataset['train'].column_names:
                dataset = dataset.rename_column('label', 'targets')
            trainset = FEDISIC2019Dataset(dataset['train'])
            testset = FEDISIC2019Dataset(dataset['test'])
        elif dataset_name == "adult":
            dataset = load_dataset("scikit-learn/adult-census-income")
            train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=self.seed)
        
            trainset = AdultCensusDataset(train_test_split['train'])
            testset = AdultCensusDataset(train_test_split['test'])
            
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported.")
        return trainset, testset

    def _partition_data(self) -> Tuple[List[Dataset], Dataset]:
        """Partition the dataset according to the specified method."""
        if self.config.name == "femnist":
            # Return pre-partitioned client datasets
            return self.trainset, self.testset
        if self.config.name == "fcube":
            # Return pre-partitioned client datasets
            return self.trainset, self.testset
        partitioning = self.config.get("partitioning", "")
        if partitioning == "dirichlet":
            alpha = self.config.get("alpha", 0.5)
            return self._partition_data_dirichlet(alpha)
        elif partitioning == "label_quantity":
            labels_per_client = self.config.get("labels_per_client", 2)
            return self._partition_data_label_quantity(labels_per_client)
        elif partitioning == "iid":
            return self._partition_data_iid()
        elif partitioning == "iid_noniid":
            similarity = self.config.get("similarity", 0.5)
            return self._partition_data_iid_noniid(similarity)
        elif partitioning == "noise":
            sigma = self.config.get("sigma", 1.0)
            return self._partition_data_noise(sigma)
        else:
            raise ValueError(f"Partitioning method {partitioning} not supported.")

    def _partition_data_dirichlet(self, alpha: float) -> Tuple[List[Dataset], Dataset]:
        """Partition according to the Dirichlet distribution."""
        min_required_samples_per_client = 10
        min_samples = 0
        prng = np.random.default_rng(self.seed)
        # Access targets from Hugging Face Dataset
        if isinstance(self.trainset, FCubeDataset):
            tmp_t = self.trainset.targets.numpy()  # Alre
        else:  # Handle PyTorch datasets
            tmp_t = self.trainset.targets
            if isinstance(tmp_t, list):
                tmp_t = np.array(tmp_t)
            if isinstance(tmp_t, torch.Tensor):
                tmp_t = tmp_t.numpy()
        num_classes = len(set(tmp_t))
        total_samples = len(tmp_t)
        idx_clients: List[List] = [[] for _ in range(self.num_clients)]
        while min_samples < min_required_samples_per_client:
            for k in range(num_classes):
                idx_k = np.where(tmp_t == k)[0]
                prng.shuffle(idx_k)
                proportions = prng.dirichlet(np.repeat(alpha, self.num_clients))
                proportions = np.array([p * (len(idx_j) < total_samples / self.num_clients) for p, idx_j in zip(proportions, idx_clients)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_k_split = np.split(idx_k, proportions)
                idx_clients = [idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)]
                min_samples = min([len(idx_j) for idx_j in idx_clients])
        trainsets_per_client = [Subset(self.trainset, idxs) for idxs in idx_clients]
        return trainsets_per_client, self.testset

    def _partition_data_label_quantity(self, labels_per_client: int) -> Tuple[List[Dataset], Dataset]:
        """Partition the data according to the number of labels per client."""
        prng = np.random.default_rng(self.seed)
        targets = self.trainset.targets
        if isinstance(targets, list):
            targets = np.array(targets)
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        num_classes = len(set(targets))
        times = [0 for _ in range(num_classes)]
        contains = []
        for i in range(self.num_clients):
            current = [i % num_classes]
            times[i % num_classes] += 1
            j = 1
            while j < labels_per_client:
                index = prng.choice(num_classes, 1)[0]
                if index not in current:
                    current.append(index)
                    times[index] += 1
                    j += 1
            contains.append(current)
        idx_clients: List[List] = [[] for _ in range(self.num_clients)]
        for i in range(num_classes):
            idx_k = np.where(targets == i)[0]
            prng.shuffle(idx_k)
            idx_k_split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(self.num_clients):
                if i in contains[j]:
                    idx_clients[j] += idx_k_split[ids].tolist()
                    ids += 1
        trainsets_per_client = [Subset(self.trainset, idxs) for idxs in idx_clients]
        return trainsets_per_client, self.testset

    def _partition_data_iid(self) -> Tuple[List[Dataset], Dataset]:
        """Partition the data in an IID manner."""
        return self._partition_data_iid_noniid(similarity=1.0)

    def _partition_data_iid_noniid(self, similarity: float) -> Tuple[List[Dataset], Dataset]:
        """Partition the data in a mixed IID and non-IID manner."""
        trainsets_per_client = []
        s_fraction = int(similarity * len(self.trainset))
        prng = np.random.default_rng(self.seed)
        idxs = prng.choice(len(self.trainset), s_fraction, replace=False)
        iid_trainset = Subset(self.trainset, idxs)
        rem_trainset = Subset(self.trainset, np.setdiff1d(np.arange(len(self.trainset)), idxs))
        all_ids = np.arange(len(iid_trainset))
        splits = np.array_split(all_ids, self.num_clients)
        for i in range(self.num_clients):
            c_ids = splits[i]
            d_ids = iid_trainset.indices[c_ids]
            trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))
        if similarity == 1.0:
            return trainsets_per_client, self.testset
        tmp_t = rem_trainset.dataset.targets
        if isinstance(tmp_t, list):
            tmp_t = np.array(tmp_t)
        if isinstance(tmp_t, torch.Tensor):
            tmp_t = tmp_t.numpy()
        targets = tmp_t[rem_trainset.indices]
        num_remaining_classes = len(set(targets))
        remaining_classes = list(set(targets))
        client_classes: List[List] = [[] for _ in range(self.num_clients)]
        times = [0 for _ in range(num_remaining_classes)]
        for i in range(self.num_clients):
            client_classes[i] = [remaining_classes[i % num_remaining_classes]]
            times[i % num_remaining_classes] += 1
            j = 1
            while j < 2:
                index = prng.choice(num_remaining_classes)
                class_t = remaining_classes[index]
                if class_t not in client_classes[i]:
                    client_classes[i].append(class_t)
                    times[index] += 1
                    j += 1
        rem_trainsets_per_client: List[List] = [[] for _ in range(self.num_clients)]
        for i in range(num_remaining_classes):
            class_t = remaining_classes[i]
            idx_k = np.where(targets == i)[0]
            prng.shuffle(idx_k)
            idx_k_split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(self.num_clients):
                if class_t in client_classes[j]:
                    act_idx = rem_trainset.indices[idx_k_split[ids]]
                    rem_trainsets_per_client[j].append(Subset(rem_trainset.dataset, act_idx))
                    ids += 1
        for i in range(self.num_clients):
            trainsets_per_client[i] = ConcatDataset([trainsets_per_client[i]] + rem_trainsets_per_client[i])
        return trainsets_per_client, self.testset
        
    def _partition_data_noise(self, sigma: float) -> Tuple[List[Dataset], Dataset]:
        """Partition data into equal parts and add increasing Gaussian noise per client."""
        prng = np.random.default_rng(self.seed)
        all_indices = np.arange(len(self.trainset))
        prng.shuffle(all_indices)
        split_indices = np.array_split(all_indices, self.num_clients)
        
        client_datasets = []
        for i, indices in enumerate(split_indices):
            subset = Subset(self.trainset, indices)
            # Compute variance as sigma * (i+1)/num_clients
            variance = sigma * (i + 1) / self.num_clients
            std = math.sqrt(variance)
            noisy_subset = NoisyDataset(subset, std)
            client_datasets.append(noisy_subset)
        return client_datasets, self.testset

    def get_dataloaders(self) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
        """Create the dataloaders to be fed into the model."""
        batch_size = self.config.get("batch_size", -1)
        batch_size_ratio = self.config.get("batch_size_ratio", None)
        trainloaders = []
        valloaders = []
        for dataset in self.datasets[0]:
            len_val = int(len(dataset) / (1 / self.val_ratio)) if self.val_ratio > 0 else 0
            lengths = [len(dataset) - len_val, len_val]
            ds_train, ds_val = random_split(dataset, lengths, torch.Generator().manual_seed(self.seed))
            if batch_size == -1 and batch_size_ratio is not None:
                batch_size = int(len(ds_train) * batch_size_ratio)
            trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=batch_size))
        return trainloaders, valloaders, DataLoader(self.datasets[1], batch_size=len(self.datasets[1]))

class NoisyDataset(Dataset):
    """Dataset wrapper to add Gaussian noise to each sample."""
    def __init__(self, subset, noise_std):
        self.subset = subset
        self.noise_std = noise_std

    def __getitem__(self, index):
        x, y = self.subset[index]
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = x + noise
        return x_noisy, y

    def __len__(self):
        return len(self.subset)

class DatasetStatistics:
    def __init__(self, fed_dataset: FederatedDataset):
        self.fed_dataset = fed_dataset
    
    def compute_statistics(self):
        stats = {}
        
        # Global dataset statistics
        trainset_size = len(self.fed_dataset.trainset)
        testset_size = len(self.fed_dataset.testset)
        stats["Total Training Samples"] = trainset_size
        stats["Total Test Samples"] = testset_size
        
        # Client-wise statistics
        client_stats = {}
        for i, client_dataset in enumerate(self.fed_dataset.datasets[0]):
            labels = [label for _, label in client_dataset]
            label_counts = dict(Counter(labels))
            client_stats[f"Client {i}"] = {
                "Num Samples": len(client_dataset),
                "Label Distribution": label_counts
            }
        stats["Client Statistics"] = client_stats
        
        # Testset statistics
        test_labels = [label for _, label in self.fed_dataset.testset]
        stats["Test Label Distribution"] = dict(Counter(test_labels))
        
        return stats
    
    def display_statistics(self):
        stats = self.compute_statistics()
        print("Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
    
    def plot_statistics(self, figsize=(20, 15)):
        """
        Generate visualizations for the dataset statistics
        
        Args:
            figsize (tuple): Figure size for the plots
        """
        stats = self.compute_statistics()
        
        # Create a figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # 1. Distribution of samples across clients
        plt.subplot(2, 2, 1)
        client_samples = [stats["Client Statistics"][f"Client {i}"]["Num Samples"] 
                          for i in range(len(stats["Client Statistics"]))]
        plt.bar(range(len(client_samples)), client_samples)
        plt.title("Sample Distribution Across Clients")
        plt.xlabel("Client ID")
        plt.ylabel("Number of Samples")
        plt.xticks(range(len(client_samples)), [f"Client {i}" for i in range(len(client_samples))])
        plt.grid(True, alpha=0.3)
        
        # 2. Label distribution in test dataset
        plt.subplot(2, 2, 2)
        test_labels = stats["Test Label Distribution"]
        labels = sorted(test_labels.keys())
        counts = [test_labels[label] for label in labels]
        plt.bar(labels, counts)
        plt.title("Test Dataset Label Distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        # 3. Heatmap of label distribution across clients
        plt.subplot(2, 1, 2)
        
        # Get all unique labels
        all_labels = set()
        for client in stats["Client Statistics"].values():
            all_labels.update(client["Label Distribution"].keys())
        all_labels = sorted(list(all_labels))
        
        # Create a matrix for the heatmap
        heatmap_data = np.zeros((len(stats["Client Statistics"]), len(all_labels)))
        
        # Fill the matrix
        for i, client_id in enumerate(range(len(stats["Client Statistics"]))):
            client_key = f"Client {client_id}"
            client_labels = stats["Client Statistics"][client_key]["Label Distribution"]
            for j, label in enumerate(all_labels):
                heatmap_data[i, j] = client_labels.get(label, 0)
        
        # Normalize by row (client)
        row_sums = heatmap_data.sum(axis=1, keepdims=True)
        heatmap_data_normalized = heatmap_data / row_sums
        
        # Plot heatmap
        sns.heatmap(heatmap_data_normalized, annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=all_labels, yticklabels=[f"Client {i}" for i in range(len(stats["Client Statistics"]))])
        plt.title("Normalized Label Distribution Across Clients")
        plt.xlabel("Label")
        plt.ylabel("Client")
        
        plt.tight_layout()
        plt.savefig("dataset_statistics.png")
        plt.show()