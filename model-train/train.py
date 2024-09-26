import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch.multiprocessing as mp

from net import Net


def train(rank, args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break

def main():
    parser = argparse.ArgumentParser(description="MNIST Training Script")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        metavar="N",
        help="how many training processes to use (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--save-dir", default="/opt/mount/model/", help="checkpoint will be saved in this directory"
    )
    parser.add_argument('--save_model', action='store_true', default=True,
                    help='save the trained model to state_dict')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')

    args = parser.parse_args()

    # Ensure the save directory exists
    if not os.path.exists('/opt/mount/model/'):
        print("Creating model directory")
        os.makedirs('/opt/mount/model/')
    else:
        print("model directory exists")  

    torch.manual_seed(args.seed)
    # create model and setup mp
    device = torch.device("cpu")


    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }

    # create mnist train dataset
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn', force=True)

    model = Net().to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here

    # mnist hogwild training process
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataset1, kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # save model checkpoint
    if args.save_model:

        PATH = "/opt/mount/model/mnist_cnn.pt"
        print(f"Storing model checkpoint {PATH}")
        torch.save(model.state_dict(), PATH)

if __name__ == "__main__":
    main()
