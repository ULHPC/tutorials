[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/bigdata/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/bigdata/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Tensorflow/Keras programming for IPU

     Copyright (c) 2024 P. Pochelu and UL HPC Team <hpc-team@uni.lu>

This programming tutorial for IPU is designed for developers looking to accelerate training and evaluation of their model, tabular and computer vision.

## Performance difference

The image below illustrates the difference of GPU (Tesla released in 2018) and IPU (released in 2017):

![IPU GPU comparison](experimental_hardware/graphcore_ipu/machine_learning/CVbench_IPUGPU.png)

**Experimental settings:**

20 neural networks from `keras.applications` package with 1 and 32 batch size values. The identical code was executed on both IPU and GPU. The speedup ratio of the IPU over the GPU is depicted on the horizontal axis, and the arithmetic intensity (FLOPS per parameters) is the vertical axis. Each data point on the plot is labeled with the corresponding neural network architecture code name along with its associated batch size. For example, "eff1_32" corresponds to EfficientNetB1 with a batch size of 32. It is important to note that both axes are represented on a logarithmic scale.

In overall, training all those tasks one-by-one during 1 epoch takes 3h14 on GPU, 1h06 minutes on IPU.


## Installation ML frameworks:

Local wheels in the ULHPC graphcore1 server: ~/poplar_sdk-ubuntu_20_04-3.0.0+1145-1b114aac3a/

* Tensorflow:
    - Tensorflow for AMD EPYC CPU: `pip install ~/poplar_sdk-ubuntu_20_04-3.0.0+1145-1b114aac3a/tensorflow-2.6.3+gc3.0.0+236842+d084e493702+amd_znver1-cp38-cp38-linux_x86_64.whl --no-index`
    - Keras front-end: `pip install ~/poplar_sdk-ubuntu_20_04-3.0.0+1145-1b114aac3a/keras-2.6.0+gc3.0.0+236851+1744557f-py2.py3-none-any.whl --no-index`
    - ipu addons for Tensorflow `pip install ~/poplar_sdk-ubuntu_20_04-3.0.0+1145-1b114aac3a/ipu_tensorflow_addons-2.6.3+gc3.0.0+236851+2e46901-py3-none-any.whl --no-index`

* JAX: `pip install jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk300 -f https://graphcore-research.github.io/jax-experimental/wheels.html`

* HuggingFace: `pip install 'transformers @ https://github.com/graphcore/transformers-fork@v4.18-gc#egg=transformers'`

* PyTorch for IPU (named "PopTorch"): https://docs.graphcore.ai/projects/pytorch-quick-start/en/latest/quick-start-beginners.html?highlight=poptorch-3.0.0%2B86945_163b7ce462_ubuntu_20_04-cp38-cp38-linux_x86_64.whl#install-the-poptorch-wheel-and-validate

## Check your ML framework installation for ipu:


After installing Jax and Tensorflow you should see your favorite frameworks and the associated IPU addons.

JAX on IPU packages:
```
jax                      0.3.16+ipu       
jax-jumpy                1.0.0            
jaxlib                   0.3.15+ipu.sdk300
```

Tensorflow on IPU packages:
```
tensorflow               2.6.3
ipu-tensorflow-addons    2.6.3
```

Poptorch on IPU packages:
```
poptorch                 3.0.0+86945
torch                    1.10.0+cpu       
torchvision              0.11.1 
```




## Tensorflow IPU code


Begin by importing Keras and TensorFlow. We'll encapsulate our training/testing loop within the 'scope' object. This object facilitates the definition of how AI accelerators (GPU or IPU) are utilized and sets the storage strategy for the model's parameters.


```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

scope = tf.distribute.get_strategy().scope # https://www.tensorflow.org/guide/distributed_training

```

Run the code below only if you use IPU. This is the only code difference between IPU and GPU implementation.


```python
##############################IPU CONTEXT###############################
from tensorflow.python import ipu

# do not directly use tensorflow.compiler but ipu_compiler if the below application is Tensorflow (and not Keras).
from tensorflow.python.ipu import ipu_compiler as compiler

# Below code is detailed in:
# https://github.com/graphcore/tensorflow/blob/r2.6/sdk-release-3.2/tensorflow/python/ipu/config.py
cfg = ipu.config.IPUConfig()  # Create the IPU hardware configure
cfg.auto_select_ipus = 1  # Attach one IPU to the current process (or MPI rank)
# TODO: other settings include FP32, FP16, ...
cfg.configure_ipu_system()  # Running hardware configuration IPU
######################################################################

```

Define the main hyperparameter and deep learning architecture


```python
BATCH_SIZE=2
IMG_SIZE=224
NUM_EPOCHS=3

def get_model():
    input_layer = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3),batch_size=BATCH_SIZE)

    x = tf.keras.applications.ResNet50(weights=None, include_top=False, classes=10)(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=input_layer, outputs=x)

    return model
```

## Read the data

Read the data, scale, and sub-samples the dataset to accelerate the demo



```python
##############################
#### READ THE DATA      ######
###############################
keras.backend.set_image_data_format("channels_last")

# Reading raw images
(trainImages, trainLabels), (
    testImages,
    testLabels,
) = keras.datasets.cifar10.load_data()

# FOR DEBUGGING PURPOSE
MAX_IMG=1000
trainImages = trainImages[:MAX_IMG]
trainLabels = trainLabels[:MAX_IMG]
testImages = testImages[:MAX_IMG]
testLabels = testLabels[:MAX_IMG]


# Preprocessing data from [0;255] to [0;1.0]
trainImages = trainImages.astype(np.float32) / 255.0
testImages = testImages.astype(np.float32) / 255.0
trainLabels = trainLabels.astype(np.int32)
testLabels = testLabels.astype(np.int32)

# Selection of images. The nunber of images should be multiple of the batch size, otherwise remainding images are ignored.
training_images = int(
    (len(trainImages) // BATCH_SIZE) * BATCH_SIZE
)  # Force all steps to be the same size
testing_images = int((len(testImages) // BATCH_SIZE) * BATCH_SIZE)
trainImages = trainImages[:training_images]
trainLabels = trainLabels[:training_images]
testImages = testImages[:testing_images]
testLabels = testLabels[:testing_images]

```

## Efficient data access

Design efficient Tensorflow I/O pipeline through asynchronous process between training and resizing images.


```python
#  Some Keras versions display wrong alert messages "INVALID_ARGUMENT" . Please ignore them
train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(trainImages, tf.float32), tf.cast(trainLabels, tf.float32))
)

eval_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(testImages, tf.float32), tf.cast(testLabels, tf.float32))
)
xy_train_gen = (
    train_dataset.shuffle(len(trainImages))
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(lambda image, label: (tf.image.resize(image, (IMG_SIZE, IMG_SIZE)), label))
    .prefetch(tf.data.AUTOTUNE)
)
xy_test_gen = (
    eval_dataset.shuffle(len(testImages))
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(lambda image, label: (tf.image.resize(image, (IMG_SIZE, IMG_SIZE)), label))
    .prefetch(tf.data.AUTOTUNE)
)

```

Buiding, training, evaluating it


```python
######################################
##### KERAS MODEL DEFINITION #########
######################################
keras.backend.set_image_data_format("channels_last")

with scope():
  model=get_model()

  model.summary()

  # Call the Adam optimizer
  if hasattr(tf.keras.optimizers, "legacy"):  # Last TF2 version
      optimizer = tf.keras.optimizers.legacy.Adam(0.01)
  else:  # Older TF2 version
      optimizer = tf.keras.optimizers.Adam(0.01)

  # Compute the number of steps
  train_steps_per_exec = len(trainImages) // BATCH_SIZE
  eval_steps_per_exec = len(testImages) // BATCH_SIZE

  ############
  # TRAINING #
  ############
  # Keras computing graph construction. Plugs together : the model, the loss, the optimizer
  model.compile(
      optimizer=optimizer,
      loss="sparse_categorical_crossentropy",
      steps_per_execution=train_steps_per_exec
  )

  model.fit(
      xy_train_gen,
      epochs=NUM_EPOCHS,
      batch_size=BATCH_SIZE
  )

  ###############
  # EVALUATING #
  ###############
  model.compile(
      metrics=["accuracy"],
      loss="sparse_categorical_crossentropy",
      steps_per_execution=eval_steps_per_exec,
  )

  (loss, accuracy) = model.evaluate(xy_test_gen, batch_size=BATCH_SIZE)

  print(f"test loss: {round(loss, 4)}, test acc: {round(accuracy, 4)}")

```

The output is:
```bash
2024-02-08 09:35:38.053169: I tensorflow/compiler/plugin/poplar/driver/poplar_platform.cc:43] Poplar version: 3.0.0 (fa83d31c56) Poplar package: 1e179b3b85
2024-02-08 09:35:39.753460: I tensorflow/compiler/plugin/poplar/driver/poplar_executor.cc:1619] TensorFlow device /device:IPU:0 attached to 1 IPU with Poplar device ID: 0

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(2, 224, 224, 3)]        0         
_________________________________________________________________
resnet50 (Functional)        (None, None, None, 2048)  23587712  
_________________________________________________________________
global_average_pooling2d (Gl (2, 2048)                 0         
_________________________________________________________________
flatten (Flatten)            (2, 2048)                 0         
_________________________________________________________________
dense (Dense)                (2, 10)                   20490     
=================================================================
Total params: 23,608,202
Trainable params: 23,555,082
Non-trainable params: 53,120
_________________________________________________________________
2024-02-08 09:35:41.557916: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/3
2024-02-08 09:35:45.304415: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:210] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
Compiling module a_inference_train_function_18311__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.19234:
[##################################################] 100% Compilation Finished [Elapsed: 00:04:04.2]
2024-02-08 09:39:53.086734: I tensorflow/compiler/jit/xla_compilation_cache.cc:376] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
50/50 [==============================] - 255s 5s/step - loss: 12.2200
Epoch 2/3
50/50 [==============================] - 2s 32ms/step - loss: 2.8860
Epoch 3/3
50/50 [==============================] - 1s 23ms/step - loss: 2.5464
Compiling module a_inference_test_function_22828__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.2593:
[##################################################] 100% Compilation Finished [Elapsed: 00:01:01.2]
50/50 [==============================] - 65s 1s/step - loss: 51.4884 - accuracy: 0.0800
test loss: 51.4884, test acc: 0.08
```

Please note the presence of the line below:

[##################################################] 100% Compilation Finished

which show the successful compilation of the computing graph on IPU. This line is not present when ran on GPU.


## Monitoring IPU activity

Don't forget `gc-monitor` and check that our process is actually running on the IPU. Illustration below:

```bash
ipuuser@graphcore1:~$ gc-monitor
+---------------+---------------------------------------------------------------------------------+
|  gc-monitor   |                Partition: p [active] has 16 reconfigurable IPUs                 |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|    IPU-M    |       Serial       |IPU-M SW|Server version|  ICU FW  | Type  | ID | IPU# |Routing|
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|10.44.44.162 | 0011.0002.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 0  |  3   |  DNC  |
|10.44.44.162 | 0011.0002.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 1  |  2   |  DNC  |
|10.44.44.162 | 0011.0001.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 2  |  1   |  DNC  |
|10.44.44.162 | 0011.0001.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 3  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|10.44.44.130 | 0022.0002.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 4  |  3   |  DNC  |
|10.44.44.130 | 0022.0002.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 5  |  2   |  DNC  |
|10.44.44.130 | 0022.0001.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 6  |  1   |  DNC  |
|10.44.44.130 | 0022.0001.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 7  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|10.44.44.226 | 0029.0002.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 8  |  3   |  DNC  |
|10.44.44.226 | 0029.0002.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 9  |  2   |  DNC  |
|10.44.44.226 | 0029.0001.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 10 |  1   |  DNC  |
|10.44.44.226 | 0029.0001.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 11 |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|10.44.44.194 | 0003.0002.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 12 |  3   |  DNC  |
|10.44.44.194 | 0003.0002.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 13 |  2   |  DNC  |
|10.44.44.194 | 0003.0001.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 14 |  1   |  DNC  |
|10.44.44.194 | 0003.0001.8222721  | 2.6.0  |    1.11.0    |  2.5.9   | M2000 | 15 |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
+-------------------------------------------------------------------------------------------------------------------------------------------+------------------------+-----------------+
|                                                     Attached processes in partition p                                                     |          IPU           |      Board      |
+--------+------------------------------------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
|  PID   |                                                  Command                                                   |  Time  |    User    | ID |  Clock   |  Temp  |  Temp  | Power  |
+--------+------------------------------------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
|1653623 |                                                  python3                                                   |  40s   |  ipuuser   | 0  | 1850MHz  | 36.3 C | 30.4 C |130.7 W |
+--------+------------------------------------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
ipuuser@graphcore1:~$
```

## Going further

* Don't use Tensorflow for Graph Neural Network, use instead PopTorch. The performance are either very slow (https://keras.io/examples/graph/gnn_citations/) or the processing process is stuck (https://graphneural.network/layers/convolution/)
* Switching beween Multi-IPU and Multi-GPU. Multi-IPU is backed by Popdist and Multi-GPU Horovod : https://github.com/PierrickPochelu/IPU_GPU_running_context


