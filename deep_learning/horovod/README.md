AI is full of computing challenges:
* Indepedant training of multiple experiments

* Training at scale 1 model (this tutorial)

* Deploying predictions

* Big Data processing at scale

# Horovod

Horovod is a framework for scaling training baed on the Ring All-Reduce protocol. Unlike approaches that use a centralized memory for aggregating and broadcasting weights' updates during stochastic gradient descent (SGD) computations, Horovod takes advantage of the computing machine's communication links, such as NVLink, to maximize performance. Horovod seamlessly integrates with popular modern deep learning frameworks like Keras2, TensorFlow2, PyTorch2, making it easy to incorporate into existing workflows.

Horovod website : https://horovod.readthedocs.io/en/stable/


By using Horovod, you can attempt to accelerate the distributed training time $$T$$ compared to the time for a singke accelerator $$G$$. However, communication can reduce scalability, and the batch size is often inversely proportional to the communication time.


The theoretical estimation for $$T$$ is $$T=G/W+C(W)$$, with $$W$$ the number of workers, and $$C$$ the communiction time between workers.




## Pre-requiste
Iit is assumed you already use a modern deep learning framework like Tensorflow2 or PyTorch2.


## Installation

ULHPC team propose either to load a previously installed or you can install it yourself. 

### CASE (1) - Loaded installation
```console
source /work/projects/ulhpc-tutorials/PS10-Horovod/env.sh
```

### CASE (2) - Install it yourself

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
1. **Initalizing Horvod**. The Horovod object it is generally named "hvd". It contains collective communication primitives and callbacks.
1. **Adaptating the local_batch** according the desired global_bath_size and the number of workers
1. **Pinning AI-accelerator** to workers in a bijective way
1. **Creating data generators** which will transfer asynchronously and efficiently data samples Disk->RAM->VRAM. The transfer between I/O to RAM is a "shard" and from the RAM to the VRAM the "local batch".
1. **Building the neural network** in each GPU VRAM
1. **Training** it
1. Evaluating it (time & accuracy)


Bonus : You can add some features (e.g, Horovod callbacks) for adding more features to your code but they come with a speed overheads. Example: verbosity, monitoring the validation metric, regular checkpointing after each epoch, learning rate scheduling with loss plateau detection, ...


## Code
[Tensorflow/Keras ULHPC example](tensorflow_horovod.py)

[Torch ULHPC example](torch_horovod.py)

[Horovod official examples](https://github.com/horovod/horovod/tree/master/examples)


## Output

[Tensorflow/Keras ULHPC example](tensorflow_horovod.py)

We launch the code with **1 GPU**

```console
(base) 255 [ppochelu@iris-195 app](3120312 1N/T/1CN)$ mpirun -n 1 python tensorflow_horovod.py
[...]
Epoch 4/4
195/195 - 147s - loss: 0.1119 - 147s/epoch - 753ms/step
Loss:  1.7116930484771729  accuracy:  0.4459134638309479
```


We launch the code with **2 GPUs**

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




Same thing with PyTorch **1 GPU**:

```console
(base) 0 [ppochelu@iris-195 app](3120453 1N/T/1CN)$ mpirun -n 1 python pytorch_horovod.py
MPI startup(): Warning: I_MPI_PMI_LIBRARY will be ignored since the hydra process manager was found
[...]
Epoch: 4 141 sec.
Loss:  -0.7153724431991577  accuracy:  0.7164999842643738
```

Same thing with PyTorch **2 GPUs**:
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
    *  Adam optimizer offers better experimental results than SGD. The adaptive nature of the Adam optimizer can help alleviate some of the convergence issues associated with LBS. (https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e)
    *  Adapting the neural network architecture for scalability. For example, some suggest that wider model can scale better: (https://proceedings.neurips.cc/paper/2018/file/e7c573c14a09b84f6b7782ce3965f335-Paper.pdf)

