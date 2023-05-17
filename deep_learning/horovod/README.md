AI is full of computing challenges:
* Indepedant training of multiple experiments
* Training at scale (this tutorial)
* Deploying predictions
* Input data processing at scale

# Horovod

Horovod is an efficient framework for scaling training that utilizes the Ring All-Reduce protocol. Unlike approaches that use a centralized memory for aggregating and broadcasting weights' updates during stochastic gradient descent (SGD) computations, Horovod takes advantage of the computing machine's communication links, such as NVLink, to maximize performance.

In the Ring All-Reduce protocol, each accelerator in the distributed system is responsible for receiving and broadcasting a portion of the gradients. This decentralized approach eliminates the bottleneck of a single accelerator handling all the communication, allowing for more efficient utilization of resources. By maximizing the communication links within a computing machine, Horovod reduces communication overhead and improves training throughput.

This approach is particularly beneficial for deep learning models that require large-scale distributed training across multiple GPUs and machines. Horovod seamlessly integrates with popular modern deep learning frameworks like Keras2, TensorFlow2, PyTorch2, making it easy to incorporate into existing workflows.

Horovod website : https://horovod.readthedocs.io/en/stable/


By using Horovod, you can attempt to accelerate the distributed training time *T* compared to 1 accelerator taking *G* seconds. 

We expect *T < G* but it is not always the case.

*T* can be theoretically estimated with *T=G/W+C*, with *W* the number of workers, and *C* the communiction time. Often, the communication reduces the scalability of the training and the batch size is inversely proportional to the  communication time.




## Pre-requiste
Iit is assumed you already use a modern one like Tensorflow2 or PyTorch2 and you are comfortable with Python too. You also have Deep Learning notions too, you know what that a batch affects training convergence, you know what is a loss function, an optimizer.


## Installation

ULHPC team propose either to load a previously installed or you can install it yourself. 

### Loaded installation
```console
source /work/projects/ulhpc-tutorials/PS10-Horovod/env.sh
```

### Install it yourself

Before installing Horovod you need to install system dependencies: MPI, CUDA, CUDNN, NCCL. All of them requires matching versions :)

There are some already installed software for helping you in the horovod installation quest:

Loading MPI and CUDA:
```console
module load toolchain/intel
```
CUDNN:
/work/projects/ulhpc-tutorials/PS10-Horovod/soft/cudnn/

NCCL:
/work/projects/ulhpc-tutorials/PS10-Horovod/soft/nccl/

Installing Horovod with NCCL:
```console
HOROVOD_GPU_OPERATIONS=NCCL`
HOROVOD_NCCL_INCLUDE=$HOROVOD_NCCL_INCLUDE
HOROVOD_NCCL_LIB=$HOROVOD_NCCL_LIB
pip install --no-cache-dir --force-reinstall horovod
```

### Checking Horovod

```console
horovodrun --check-build
```

Should produce

```console
Available Frameworks:
    [X] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [X] Gloo
```

Check if your deep learning framework is checked, MPI and NCCL.

## HPC/AI methodology

To assess how your workload can benefit from using Horovod, it is recommended to perform one of the following analyses:
* **Strong Scalability Analysis**. It consists in fixing the AI workload behaviour (Batch size, model, ...) and scaling the number of machine. By increasing the computational resources while maintaining the other workload characteristics, you can observe the impact of scaling on the training speed.
* **Weak Scalability Analysis**. It consists in fixing a number of AI-accelerator (e.g, 4) and scale a characteristic of the AI workload, such as the number of layers. This analysis helps understand how a parameter affects the accuracy, scalability and speed of the workload when varying a specific aspect.

## Horovod typical code

The proposed codes contains those 7 block of codes
1. Initalizing the Horvod object (containing communication primitives)
1. Adaptating the batch to the global_bath_size and the number of workers
1. Pinning AI-accelerator to workers
1. Creating generators which will transfer asynchronously and efficiently data samples Disk->RAM->VRAM. The transfer between I/O to RAM is refered as "shard" and from the RAM to the VRAM the "local batch".
1. Building the Neural Network
1. Training it
1. Evaluating it (time & accuracy)


Bonus : You can add some “hooks” (or “callbacks”) for adding features to your code but they come with a speed overheads: including verbosity, regular validation metric computing, regular checkpointing, learning rate scheduling with loss plateau detection, … 

## Code
[Tensorflow/Keras ULHPC example](app/tensorflow_horovod.py)

[Torch ULHPC example](app/torch_horovod.py)

[Horovod official examples](https://github.com/horovod/horovod/tree/master/examples)


## Output


We launch tensorflow2 code with **1 GPU**

```console
(base) 255 [ppochelu@iris-195 app](3120312 1N/T/1CN)$ mpirun -n 1 python tensorflow_horovod.py
[...]
Epoch 4/4
195/195 - 147s - loss: 0.1119 - 147s/epoch - 753ms/step
Loss:  1.7116930484771729  accuracy:  0.4459134638309479
```


Now with **2 GPUs**

```console
(base) 130 [ppochelu@iris-192 app](3120466 1N/T/1CN)$ mpirun -n 2 python tensorflow_horovod.py
[...]
Epoch 4/4
Epoch 4/4
195/195 - 92s - loss: 0.1164 - 92s/epoch - 472ms/step
195/195 - 92s - loss: 0.1173 - 92s/epoch - 469ms/step
Loss:  1.3920882940292358  accuracy:  0.5380609035491943
Loss:  1.3958407640457153  accuracy:  0.5354567170143127
```




Now PyTorch code with **1 GPU**:

```console
(base) 0 [ppochelu@iris-195 app](3120453 1N/T/1CN)$ mpirun -n 1 python pytorch_horovod.py
MPI startup(): Warning: I_MPI_PMI_LIBRARY will be ignored since the hydra process manager was found
[...]
Epoch: 4 141 sec.
Loss:  -0.7153724431991577  accuracy:  0.7164999842643738
```

Now **2 GPUs**:
```console
base) 0 [ppochelu@iris-195 app](3120453 1N/T/1CN)$ mpirun -n 2 python pytorch_horovod.py
[...]
Epoch: 4 85 sec.
Loss:  -0.6600856781005859  accuracy:  0.6620000004768372
Epoch: 4 85 sec.
Loss:  -0.6600856781005859  accuracy:  0.6620000004768372
```

We see the prediction quality are similar around 70% +/- 4% but 2 GPUs is 141/85=1.65 times faster.

## Going further towards scalability

Bigger batch reduce the communication need. If your are facing scalability issue increase the batch size.

* Large Batch Size (LBS) such as >1024 may hurts the convergence, for mitigating this: 
    *  Learning Rate scheduling. This can help compensate for the challenges posed by larger batch sizes and aid in achieving better convergence.
    *  Adam optimizer offers better experimental results than SGD. The adaptive nature of the Adam optimizer can help alleviate some of the convergence issues associated with LBS. https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e
    *  Re-thinking the neural network architecture for scalability https://proceedings.neurips.cc/paper/2018/file/e7c573c14a09b84f6b7782ce3965f335-Paper.pdf

