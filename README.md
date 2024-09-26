[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/H1dh0F7f)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=16009686&assignment_repo_type=AssignmentRepo)
# EMLO4 - Session 03

# Docker Compose - MNIST Training, Evaluation, and Inference Services

## Overview
This repository provides a Docker-based setup for training, evaluating, and running inference on the `MNIST` dataset using a Convolutional Neural Network (CNN) in PyTorch. It includes three main services:

- Train - To train the CNN model on the `MNIST` dataset.
- Evaluate - To evaluate the trained model on the test set.
- Infer - To perform inference on a few random samples from the `MNIST` test set.

The model and training technique used in this is taken from (MNIST Hogwild): https://github.com/pytorch/examples/tree/main/mnist_hogwild
and the number of processes is set to 2.

The services are defined using Docker Compose, which allows you to spin up isolated containers for each task, ensuring reproducibility and an easy-to-deploy environment.

## Project Structure

```
.
├── data/                # Local data directory (mounted to Docker containers)
├── model-train/         # Directory for training Docker service
│   ├── Dockerfile.train
│   ├── train.py
│   └── net.py           # Common neural network architecture
├── model-eval/          # Directory for evaluation Docker service
│   ├── Dockerfile.eval
│   ├── evaluate.py
│   └── net.py
├── model-infer/         # Directory for inference Docker service
│   ├── Dockerfile.infer
│   ├── infer.py
│   └── net.py
├── docker-compose.yml   # Docker Compose configuration file
└── README.md            # This README file

```

## Prerequisites
Before you start, make sure you have the following installed on your machine:

- Docker
- Docker Compose

## How to Run
#### 1. Clone the Repository
First, clone this repository to your local machine:

```
git clone https://github.com/your-repo/mnist-docker.git
cd mnist-docker
```

#### 2. Prepare the Dataset
Download the `MNIST` dataset during training. The dataset will be stored in the `./data` directory, which is mounted into the Docker containers. This directory ensures that the same data is shared across all services.

#### 3. Build and Start Services
Use Docker Compose to build and run the services:


```
docker-compose up --build
```

or 

```
docker compose build
```

This command will build the `train`, `evaluate`, and `infer` services and begin running them in sequence. The `train` service will complete its training first, then the `evaluate` service will start to evaluate the model, followed by the `infer` service, which will generate predictions on a subset of test samples.

#### 4. Training the Model
The `train` service is responsible for training the model on the `MNIST` dataset. It uses the `train.py` script, which implements multi-processing (Hogwild training) to parallelize training on the CPU.

The trained model is saved to the `/model/mnist_cnn.pt` file, which is located in the mounted `mnist` volume, making it accessible to other services (evaluation and inference).

##### Running the Train Service

Run the Docker Compose service using

```
docker-compose run train
```
or 


```
docker compose run train
```

The training logs will be printed to the console, showing the progress of the model as it trains over the `MNIST` dataset.

#### 5. Evaluating the Model
Once the model is trained, the evaluate service evaluates the model on the `MNIST` test set using the evaluate.py script. The results are saved to a `JSON` file located at `/model/eval_results.json`.

##### Running the Evaluate Service

Run the Docker Compose service using 

```
docker-compose run evaluate
```
or 

```
docker compose run evaluate
```

After evaluation, the accuracy and loss will be printed to the console and saved to the evaluation `JSON` file.

#### 6. Running Inference
The infer service uses the infer.py script to run inference on a few random samples from the `MNIST` test set. The predictions and corresponding 5 sample images are saved in the `/model/results` directory as `PNG` files.

##### Running the Infer Service

Run the Docker Compose service using

```
docker-compose run infer
```

or


```
docker compose run infer
```

The predicted results will be saved as PNG images with filenames representing the predicted class and the sample index (e.g., 3_12.png).

## Docker Compose Configuration

The `docker-compose.yml` file orchestrates three distinct services for the `MNIST` project: `train`, `evaluate`, and `infer`. Each service runs in its own Docker container, sharing a common dataset and model directory through mounted volumes. Here’s a detailed explanation of how each section works:

### Services Overview
##### Train Service (`train`)

The train service is responsible for training the CNN model on the `MNIST` dataset. The build context points to the `model-train` directory, which contains the Dockerfile (`Dockerfile.train`) and the necessary scripts for training. The dataset directory (`./data`) is mounted into `/opt/mount/` data inside the container.

A named volume `mnist` is mounted to `/opt/mount/model` to store the trained model weights. This ensures that the model can be shared with other services.

##### Evaluate Service (`evaluate`)

The `evaluate` service loads the model trained by the `train` service and evaluates it on the `MNIST` test dataset. The build context points to the `model-eval` directory, which includes `Dockerfile.eval` and the evaluation scripts. It shares the same volumes as the train service and ensures the test dataset is accessible. The `evaluate` service depends on the `train` service. This ensures that the model is trained before evaluation starts.

##### Infer Service (`infer`)

The `infer` service uses the trained model to make predictions on random samples from the `MNIST` test dataset. The build context points to the `model-infer` directory, which includes `Dockerfile.infer` and inference scripts. Similar to the other services, the infer service mounts the dataset to access the dataset and loads the trained model from the `mnist` volume.

**Network Mode:** The service runs in `"host"` mode, which provides the container direct access to the host machine's network. This is useful for faster local communication during inference. 

The `infer` service also depends on the train service, ensuring that the model is trained before inference begins.
**Restart Policy:** The restart policy is set to no, meaning that this service will not automatically restart if it stops or encounters an error.

#### Volumes
`mnist:` This is a named volume used by all services to store the trained model (`mnist_cnn.pt`). By using this shared volume, the trained model can be accessed by the evaluate and infer services without needing to retrain it.

### Dockerfile Explanations
Each service has its own Dockerfile, which is built upon a base image that includes Python and PyTorch.

#### Training Dockerfile (`Dockerfile.train`)
This Dockerfile sets up the environment for training the model, including copying the necessary scripts and creating the output directory for storing the trained model.

#### Evaluation Dockerfile (`Dockerfile.eval`)
This Dockerfile is similar to the training one, but it focuses on evaluating the model using the trained weights saved from the training process.

#### Inference Dockerfile (`Dockerfile.infer`)
The inference Dockerfile installs any additional dependencies (like requests) needed for inference tasks, loads the trained model, and performs predictions.

### Model Architecture
The neural network (`net.py`) consists of:

- Two convolutional layers (Conv2d), each followed by a max-pooling operation.
- A dropout layer to prevent overfitting.
- Two fully connected layers (Linear), with the final output being a 10-class classification using log_softmax.

## Conclusion
This setup provides a reproducible environment for training, evaluating, and running inference on the `MNIST` dataset using Docker and PyTorch. By leveraging Docker, you can ensure consistent results across different systems, and the modular design allows for easy scaling or modifications. The Docker Compose configuration defines three services—train, evaluate, and infer—that work together to handle the complete lifecycle of model training, evaluation, and inference. Each service is isolated in its own container but shares common resources like the dataset and model through volumes. The train service trains the model, the evaluate service evaluates the trained model, and the infer service performs predictions on random test samples. By using Docker Compose, these services are containerized, making the setup reproducible and portable across different environments.