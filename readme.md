# Federated Learning with ResNet9 on CIFAR-10

This repository contains an experimental implementation of Federated Learning using the ResNet9 model on the CIFAR-10 dataset. The project is inspired by the work "Improving Generalization in Federated Learning by Seeking Flat Minima" and is based on the original implementation found at:

> Caldarola, D., Caputo, B., & Ciccone, M. [_Improving Generalization in Federated Learning by Seeking Flat Minima_](https://arxiv.org/abs/2203.11834), _European Conference on Computer Vision_ (ECCV) 2022.

## Overview
This project explores federated learning techniques with an emphasis on optimizing generalization by leveraging federated averaging (FedAvg) and federated optimization (FedOpt). The experiment runs a ResNet9 model on CIFAR-10 with various hyperparameters to analyze the impact of different configurations.

## Setup
### Environment Setup
#### Using Conda (Preferred)
```bash
conda env create -f environment.yml
```
#### Using Pip (Alternative)
```bash
pip3 install -r requirements.txt
```

### Dataset
The CIFAR-10 dataset is automatically downloaded and preprocessed when running the training script.
```
conda activate torch10
cd data
chmod +x setup_datasets.sh
./setup_datasets.sh
```

## Running Experiments
Use the following debug configuration in `launch.json` to execute the training process in Visual Studio Code:

```json
{
    "name": "Run FedAvg (alpha=0.05)",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/models/main.py",
    "cwd": "${workspaceFolder}/models",
    "console": "integratedTerminal",
    "args": [
        "-dataset", "cifar10",
        "--num-rounds", "10000",
        "--eval-every", "10",
        "--batch-size", "1024",
        "--num-epochs", "1",
        "--clients-per-round", "5",
        "-model", "resnet9",
        "-lr", "0.01",
        "--weight-decay", "0.0004",
        "-device", "mps",
        "-algorithm", "fedopt",
        "--server-lr", "1",
        "--server-opt", "sgd",
        "--num-workers", "2",
        "--where-loading", "init",
        "-alpha", "0.05"
    ]
}
```

## Experiment Details
- **Model:** ResNet9
- **Dataset:** CIFAR-10
- **Federated Learning Algorithm:** FedOpt
- **Number of Rounds:** 10,000
- **Batch Size:** 1024
- **Epochs per Client:** 1
- **Clients per Round:** 5
- **Learning Rate:** 0.01
- **Server Optimizer:** SGD
- **Alpha Value:** 0.05 (Dirichlet distribution parameter for data partitioning)

## Experiment Results

You can view the detailed results of the Federated Learning experiments conducted in this project by following the link below:

[View Experiment Results on Weights & Biases](https://wandb.ai/nguyendam5555/Federated%20learning/runs/4ike5us5/workspace?nw=nwusernguyendam5555)

This link will direct you to the experiment's workspace on Weights & Biases, where you can explore various metrics, visualizations, and logs generated during the training process.

## Checkpoints

The checkpoints of the Federated Learning experiments are available for review. You can access them through the following link:

[View Checkpoints on Google Drive](https://drive.google.com/drive/folders/1am6jhN0h9AVQeArGtt5RKh7GGNoesmei?usp=sharing)

This link will take you to the Google Drive folder containing the model checkpoints, allowing you to track the progress and performance of the model at different stages during the training process.


## Reference
This project is inspired by and extends the work presented in the following paper:

```text
@inproceedings{caldarola2022improving,
  title={Improving generalization in federated learning by seeking flat minima},
  author={Caldarola, Debora and Caputo, Barbara and Ciccone, Marco},
  booktitle={European Conference on Computer Vision},
  pages={654--672},
  year={2022},
  organization={Springer}
}
```

For further details, please refer to the original repository: [GitHub Repository](https://github.com/debcaldarola/fedsam).

