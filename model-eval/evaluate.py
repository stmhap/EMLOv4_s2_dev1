import os
import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from net import Net


def test_epoch(model, device, data_loader):
    # write code to test this epoch
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    out = {"Test loss": test_loss, "Accuracy": accuracy}
    print(out)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))    
    return out

def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-dir", default="/opt/mount/model", help="checkpoint will be saved in this directory"
    )
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    device = torch.device("cpu")
    # create MNIST test dataset and loader
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    # create model and load state dict
    model = Net().to(device)

    if os.path.isfile('/opt/mount/model/mnist_cnn.pt'):
        print("=> Loading checkpoint 'mnist_cnn.pt'")
        model.load_state_dict(torch.load('/opt/mount/model/mnist_cnn.pt'))
    else:
        print("=> No checkpoint found at 'mnist_cnn.pt'")

    # test epoch function call
    eval_results = test_epoch(model, device, test_loader)

    with (Path(args.save_dir) / "eval_results.json").open("w") as f:
        # print(f)
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
