# Federated Learning Benchmark Framework

<p align="center">
  <a href="https://nechbamohammed.github.io/FLBenchmarkUI/">
    <img src="logo.png" width="140px" alt="FedBench Website" />
  </a>
</p>
<p align="center">
    <a href="https://nechbamohammed.github.io/FLBenchmarkUI/">Website</a> |
    <a href="">Paper</a> 
    <br /><br />
</p>


A strategic framework for systematically benchmarking federated learning algorithms across diverse datasets and client distributions. Built for researchers and practitioners to make informed algorithm selection decisions based on rigorous comparative analysis.

## Strategic Overview

FLBenchmark provides a comprehensive ecosystem for evaluating federated learning strategies in realistic scenarios. Our framework enables:

- **Strategic Algorithm Selection**: Identify optimal algorithms for specific data distributions and application constraints
- **Performance Trade-off Analysis**: Evaluate critical trade-offs between accuracy, communication efficiency, and privacy preservation
- **Heterogeneity Impact Assessment**: Measure how different heterogeneity factors affect algorithm performance
- **Systematic Comparison Methodology**: Ensure fair, reproducible comparisons between federated optimization approaches

## Installation
```python
pip install fedbench==0.1.1
```
## Quick Start Tutorials
0. **The statistics of datasets?**

   [![Kaggle](https://img.shields.io/badge/Kaggle-Dataset_Statistics-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/nechbamohammed/the-statistics-of-datasets)(or open the [Jupyter Notebook](https://github.com/NechbaMohammed/FLBenchmark/blob/main/notebooks/the-statistics-of-datasets.ipynb))



### Strategic Dataset Selection
Choose datasets that match your strategic evaluation needs:

| Dataset       | Classes | Partitioning Method                                      | Partition Settings                        |
|--------------|----------|--------------------------------------------------------|-------------------------------------------|
| MNIST, FMNIST, SVHN, CINIC-10, CIFAR-10 | 10       | label_quantity, dirichlet, iid_noniid, noise, iid      | labels_per_client, alpha, similarity, segma |
| FedISIC2019  | 8        | label_quantity, dirichlet, iid_noniid, noise, iid      | labels_per_client, alpha, similarity, segma |
| CIFAR-100    | 100      | label_quantity, dirichlet, iid_noniid, noise, iid      | labels_per_client, alpha, similarity, segma |
| Adult        | 2        | label_quantity, dirichlet, iid_noniid, iid            | labels_per_client, alpha, similarity      |
| FCUBE        | 2        | synthetic                                             | -                                         |
| FEMNIST      | 62       | real-world                                           | -                                         |


#### Dataset Usage Examples
Below are examples demonstrating how to use the FLBenchmark dataset module.
```python
from fedbench.datasets import FederatedDataset
from omegaconf import DictConfig

# Example 1: MNIST Dataset

data_config = DictConfig({"name": "mnist", "partitioning": "iid", "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=1)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
federated_dataset.print_dataset_stats()
```
```
Number of training instances: 60000
Number of test instances: 10000
Number of features: 784
Number of classes: 10
```
---

```python
# Example 2: Adult Dataset

data_config = DictConfig({"name": "adult", "partitioning": "synthetic", "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=1)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
federated_dataset.print_dataset_stats()
```
```
Number of training instances: 26048
Number of test instances: 6513
Number of features: 99
Number of classes: 2
```

---

```python
# Example 3: FEMNIST Dataset

data_config = DictConfig({"name": "femnist", "partitioning": "real-world", "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=1)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
federated_dataset.print_dataset_stats()
```
```
Number of training instances: 649184
Number of test instances: 165093
Number of features: 784
Number of classes: 62
```

```python
# Example 5: Label Quantity Partitiin

data_config = DictConfig({"name": "mnist", "partitioning": "label_quantity","labels_per_client":2, "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=10)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
```
```python
# Example 6: Dirichlet Partitiin

data_config = DictConfig({"name": "mnist", "partitioning": "dirichlet","alpha":0.5, "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=10)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
```
```python
# Example 7: Quantity Skew Partitiin

data_config = DictConfig({"name": "mnist", "partitioning": "iid_noniid","similarity":0.5, "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=10)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
```
```python
# Example 8: Feature Distribution Partitiin

data_config = DictConfig({"name": "mnist", "partitioning": "noise","segma":0.1, "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=10)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
```

### Strategic Algorithm 
#### 🧠 Model Configurations
FedBench provides pre-configured models for various datasets:

| Dataset        | Model Type                        | Input Dim | Hidden Dims       | Num Classes | MOON Variant                        |
|---------------|----------------------------------|-----------|-------------------|-------------|-------------------------------------|
| MNIST, FMNIST, FEMNIST | MNISTModel                        | 256       | [120, 84]         | 10 (MNIST, FMNIST), 62 (FEMNIST) | MnistModelMOON                    |
| CIFAR-10, SVHN, CINIC-10, FedISIC2019 | CNN                               | 400       | [120, 84]         | 10          | CNNModelMOON                       |
| CIFAR-100      | CNN                               | 400       | [120, 84]         | 100         | CNNModelMOON                       |
| Adult         | MLP                               | 99        | [32, 16, 8]       | 2           | MLPModelMOON                       |
| FCUBE         | MLP                               | 3         | [32, 16, 8]       | 2           | MLPModelMOON                       |

#### 📖 Usage Examples

##### FedAvg on MNIST
```python
from omegaconf import OmegaConf,DictConfig
import torch
from fedbench.algorithms.fedavg.simulation import run_fedavg

model_cfg = OmegaConf.create({
    "_target_": "fedbench.models.MNISTModel",
    "input_dim": 256,
    "hidden_dims": [120, 84],
    "num_classes": 10,
})

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}

data_config = DictConfig({
    "name": "mnist",
    "partitioning": "iid",
    "batch_size": 64,
})

history = run_fedavg(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=10,
    num_rounds=20,
    num_epochs=15,
    learning_rate=0.01,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
##### FedProx on CIFAR-10

```python
from omegaconf import OmegaConf,DictConfig
import torch
from fedbench.algorithms.fedprox.simulation import run_fedprox

model_cfg = OmegaConf.create({
    "_target_": "fedbench.models.CNN",
    "input_dim": 400,
    "hidden_dims": [120, 84],
    "num_classes": 10,
})

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}

data_config = DictConfig({
    "name": "cifar10",
    "partitioning": "dirichlet",
    "alpha":0.5,
    "batch_size": 32,
})

history = run_fedprox(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=20,
    num_rounds=30,
    num_epochs=10,
    learning_rate=0.005,
    mu=0.01,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

##### FedAdam on CIFAR-100

```python
from omegaconf import OmegaConf,DictConfig
import torch
from fedbench.algorithms.fedadam.simulation import run_fedadam

model_cfg = OmegaConf.create({
            "_target_": "fedbench.models.CNN",
            "input_dim": 400,
            "hidden_dims": [120, 84],
            "num_classes": 100,
        })

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}

data_config = DictConfig({
    "name": "cifar100",
    "partitioning": "iid",
    "batch_size": 32,
})

history = run_fedadam(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=100,
    num_rounds=25,
    num_epochs=20,
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
##### FedAdagrad on SVHN

```python
from omegaconf import OmegaConf,DictConfig
import torch
from fedbench.algorithms.fedadagrad.simulation import run_fedadagrad

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}
 model_cfg = OmegaConf.create({
            "_target_": "fedbench.models.CNN",
            "input_dim": 400,
            "hidden_dims": [120, 84],
            "num_classes": 10,
        })
data_config = DictConfig({
    "name": "svhn",
    "partitioning": "noise",
    "segma":0.1,
    "batch_size": 32,
})

history = run_fedadagrad(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=10,
    num_rounds=25,
    num_epochs=20,
    learning_rate=0.001,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
##### FedYogi on CINIC-10
```python
from omegaconf import OmegaConf,DictConfig
import torch
from fedbench.algorithms.fedyogi.simulation import run_fedyogi

model_cfg = OmegaConf.create({
            "_target_": "fedbench.models.CNN",
            "input_dim": 400,
            "hidden_dims": [120, 84],
            "num_classes": 10,
        })

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}
data_config = DictConfig({
    "name": "cinic10",
    "partitioning": "iid",
    "batch_size": 32,
})

history = run_fedyogi(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=10,
    num_rounds=25,
    num_epochs=20,
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
##### FedNova on FCUBE

```python
from fedbench.algorithms.fednova.simulation import run_fednova
from omegaconf import OmegaConf,DictConfig
import torch

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}

model_cfg = OmegaConf.create({
            "_target_": "fedbench.models.MLP",
            "input_dim": 3,
            "hidden_dims" : [32, 16, 8] ,
            "num_classes": 2,
        })
data_config = DictConfig({
            "name": "fcube",
            "partitioning": "synthetic",
            "batch_size": 64,
        })
history = run_fednova(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=15,
    num_rounds=25,
    num_epochs=20,
    learning_rate=0.001,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
##### Scaffold on FedISIC2019
```python

from fedbench.algorithms.scaffold.simulation import run_scaffold
from omegaconf import OmegaConf,DictConfig
import torch

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}
model_cfg = OmegaConf.create({
            "_target_": "fedbench.models.CNN" ,
            "input_dim": 400,
            "hidden_dims": [120, 84],
            "num_classes": 10,
        })
data_config = DictConfig({
    "name": "fedisic2019",
    "partitioning": "iid_noniid",
    "similarity":0.5,
    "batch_size": 32,
})


history = run_scaffold(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=15,
    num_rounds=25,
    num_epochs=20,
    learning_rate=0.001,
    model_dir="weights",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
##### MOON on Adult

```python
from fedbench.algorithms.moon.simulation import run_moon
from omegaconf import OmegaConf,DictConfig
import torch

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}

model_cfg = OmegaConf.create({
            "_target_": "fedbench.models.MLPModelMOON",
            "input_dim": 99,
            "hidden_dims" : [32, 16, 8] ,
            "output_dim":256,
            "num_classes": 2,
})

data_config = DictConfig({
    "name": "adult",
    "partitioning": "iid",
    "batch_size": 64,
})

history = run_moon(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=10,
    num_rounds=25,
    num_epochs=20,
    learning_rate=0.001,
    model_dir="weights",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
##### FedBN on FEMNIST

```python
from fedbench.algorithms.fedbn.simulation import run_fedbn
from omegaconf import OmegaConf,DictConfig
import torch

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}
 model_cfg = OmegaConf.create({
            "_target_": "fedbench.models.MNISTModel",
            "input_dim": 256,
            "hidden_dims": [120, 84],
            "num_classes": 62,
        })
data_config = DictConfig({
    "name": "femnist",
    "partitioning": "real-world",
    "batch_size": 32,
})


history = run_fedbn(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=15,
    num_rounds=25,
    num_epochs=20,
    learning_rate=0.001,
     save_path="weights",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
## 🧪 Benchmarking
You can easily benchmark different algorithms and configurations
```bash
# Clone the repository
git clone git@github.com:NechbaMohammed/FLBenchmark.git
cd FLBenchmark
# Install dependencies
pip install -r requirements.txt
```
``` python
# Run all benchmarks
python benchmark_runner.py

# Aggregate results
python aggregate_experiments.py

# Generate plots
python learning_curve_plots.py
python local_epoch_comparison_plots.py
```



## Cit📚 Citation
If you use FedBench in your research, please cite our paper:ation

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```


## 🤝 Contributing 

We welcome contributions! Please check out our contribution guidelines for details.
