import os
import json
import time
import random
import torch
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from net import Net


def infer(model, dataset, save_dir, num_samples=5):
    model.eval()
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image, _ = dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True).item()

        img = Image.fromarray(image.squeeze().numpy() * 255).convert("L")
        img.save(results_dir / f"{pred}_{idx}.png")


def main():
    save_dir = "/opt/mount/model/"

    device = torch.device("cpu")    
    # init model and load checkpoint here
    model = Net().to(device)
    if os.path.isfile('/opt/mount/model/mnist_cnn.pt'):
        print("=> Loading checkpoint 'mnist_cnn.pt'")
        model.load_state_dict(torch.load('/opt/mount/model/mnist_cnn.pt'))
    else:
        print("=> No checkpoint found at 'mnist_cnn.pt'")

	# create transforms and test dataset for mnist
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST('./data', train=False,
                       transform=transform)
    infer(model, dataset, save_dir)
    print("Inference completed. Results saved in the 'results' folder.")


if __name__ == "__main__":
    main()
