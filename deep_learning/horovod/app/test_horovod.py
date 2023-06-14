FRAMEWORK="TF" #TF or TORCH


if FRAMEWORK=="TF":
    import tensorflow as tf
    import horovod.keras as hvd
    print("List of TF visible physical GPUs : ", tf.config.experimental.list_physical_devices("GPU"))
elif FRAMEWORK=="TORCH":
    import torch
    import horovod.torch as hvd
    print("List of PyTorch visible physical GPUs : ", torch.cuda.device_count())
else:
    raise ValueError(f"FRAMEWORK '{FRAMEWORK}' NOT UNDERSTODD")

hvd.init()

msg = f"MPI_size = {hvd.size()}, MPI_rank = {hvd.rank()}, MPI_local_size = {hvd.local_size()},  MPI_local_rank = {hvd.local_rank()}"

try:
    import platform #
    msg += f" platform = {platform.node()}"
except ImportError:
    pass

print(msg)


