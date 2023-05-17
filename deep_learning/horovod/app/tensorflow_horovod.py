import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# Inform tensorflow that images will be HWC format
tf.keras.backend.set_image_data_format("channels_last")

import horovod
import horovod.tensorflow.keras as hvd

import numpy as np
import time  # For measuring computing time

BATCH_SIZE = 128
LEARNING_RATE = 0.001

tf.keras.utils.set_random_seed(42)

# Horovod: initialize Horovod.
hvd.init()

######################
# ADAPT SGD SETTINGS #
######################
local_batch_size = BATCH_SIZE // int(hvd.size())

############################################################################
# Horovod: pin GPU to be used to process local rank (one GPU per process) #
###########################################################################
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

#####################
# Reading the data #
####################
# (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
if int(hvd.size()) > 1:
    num_train_per_replica = len(X_train) // int(hvd.size())
    X_train = X_train[
        int(hvd.rank())
        * num_train_per_replica : (int(hvd.rank()) + 1)
        * num_train_per_replica
    ]
    Y_train = Y_train[
        int(hvd.rank())
        * num_train_per_replica : (int(hvd.rank()) + 1)
        * num_train_per_replica
    ]
    num_eval_per_replica = len(X_test) // int(hvd.size())
    X_test = X_test[
        int(hvd.rank())
        * num_eval_per_replica : (int(hvd.rank()) + 1)
        * num_eval_per_replica
    ]
    Y_test = Y_test[
        int(hvd.rank())
        * num_eval_per_replica : (int(hvd.rank()) + 1)
        * num_eval_per_replica
    ]
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)


print("Train Images array: ", X_train.shape)
print("Train Labels array: ", Y_train.shape)
print("Test Images array: ", X_test.shape)
print("Test Labels array: ", Y_test.shape)


def create_dataset(X, Y):
    # WARNING: display warning message "INVALID_ARGUMENT" . Those messages are Keras error messages .
    train_gen = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X, tf.float32), tf.cast(Y, tf.float32))
    )

    train_gen = (
        train_gen.shuffle(1024)
        .batch(local_batch_size, drop_remainder=True)
        .map(lambda image, label: (tf.image.resize(image, (224, 224)), label))
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_gen


train_dataset = create_dataset(X_train, Y_train)
test_dataset = create_dataset(X_test, Y_test)

####################################
# BUILDING THE DEEP LEARNING MODEL #
####################################
from tensorflow.keras.applications import ResNet50

input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
x = ResNet50(weights=None, include_top=False, classes=10)(input_layer)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs=input_layer, outputs=x)


# Horovod: adjust learning rate based on number of GPUs.
optimizer = tf.optimizers.Adam(LEARNING_RATE)

# Horovod: add Horovod DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, backward_passes_per_step=1, average_aggregated_gradients=True
)


callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes
    hvd.callbacks.BroadcastGlobalVariablesCallback(0)  # ,
    # Horovod: average metrics among workers at the end of every epoch.
    # hvd.callbacks.MetricAverageCallback(),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
# if hvd.rank() == 0:
#    callbacks.append(tf.keras.callbacks.ModelCheckpoint("./last_checkpoint.h5"))

#############
# TRAINING  #
#############
train_steps_per_exec = len(X_train) // local_batch_size
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    steps_per_execution=train_steps_per_exec,
)
hvd.allgather(tf.Variable([0]))  # synch. barrier before starting the timer

model.fit(
    train_dataset,
    batch_size=local_batch_size,
    callbacks=callbacks,
    epochs=4,
    verbose=2,  # verbose=0 -> hiding , verbose=1 -> display
)
##############
# EVALUATING #
##############
eval_steps_per_exec = len(X_test) // local_batch_size
model.compile(
    metrics=["accuracy"],
    loss="sparse_categorical_crossentropy",
    steps_per_execution=eval_steps_per_exec,
)

(loss, accuracy) = model.evaluate(
    test_dataset,
    batch_size=local_batch_size,
    steps=eval_steps_per_exec,
    callbacks=callbacks,
    verbose=0,
)
print("Loss: ", loss, " accuracy: ", accuracy)

