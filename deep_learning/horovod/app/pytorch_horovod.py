import argparse
import os

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
from torchvision import datasets, transforms

import horovod
import horovod.torch as hvd

import time

import warnings

warnings.filterwarnings("ignore")

BATCH_SIZE = 256
LEARNING_RATE = 0.001

torch.manual_seed(42)

# Horovod initialize
torch.cuda.empty_cache()
hvd.init()

######################
# ADAPT SGD SETTINGS #
######################
local_batch_size = BATCH_SIZE // int(hvd.size())

############################################################################
# Horovod: pin GPU to be used to process local rank (one GPU per process) #
###########################################################################
if torch.cuda.is_available():
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(42)
    kwargs = {"num_workers": 1, "pin_memory": True}
else:
    kwargs = {}
# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(1)


#########################################
# Reading the data and create generator #
#########################################
def get_cifar10_dataset(is_training, local_batch_size):
    torch_dataset = datasets.CIFAR10(
        "./data/",
        train=is_training,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        ),
    )  # already scaled between [0;1]

    # Horovod: use DistributedSampler to partition the training data.
    torch_sampler = torch.utils.data.distributed.DistributedSampler(
        torch_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    torch_loader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=local_batch_size, sampler=torch_sampler, **kwargs
    )
    return torch_loader


train_loader = get_cifar10_dataset(True, local_batch_size)
test_loader = get_cifar10_dataset(False, 16)


####################################
# BUILDING THE DEEP LEARNING MODEL #
####################################
import torch
import torchvision.models as models

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, 10), torch.nn.Softmax(dim=1))


if torch.cuda.is_available():
    model.cuda()  # Move model to GPU.


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
    op=hvd.Average,
    gradient_predivide_factor=1,
)


##############################
# TRAINING AND EVALUATING IT #
##############################
def train_epoch(epoch, train_loader, optimizer, model):
    model.train()
    print("Train Epoch: ", epoch)
    train_loader.sampler.set_epoch(epoch)  # Set epoch for shuffling.
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print(loss)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(test_loader, model):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_loader.sampler)
    test_accuracy /= len(test_loader.sampler)

    # Horovod: average metric values across workers.
    loss = metric_average(test_loss, "avg_loss")
    accuracy = metric_average(test_accuracy, "avg_accuracy")

    return loss, accuracy


hvd.allgather(torch.tensor([0]))
start_time = time.time()
for epoch in range(4):
    train_epoch(epoch, train_loader, optimizer, model)
print("Enlapsed training time: ", round(time.time() - start_time), " sec")

(loss, accuracy) = test(test_loader, model)
print("Loss: ", loss, " accuracy: ", accuracy)

