
Copyright (c) 2023 P. Pochelu <pierrick.pochelu@uni.lu> and UL HPC Team  <hpc-sysadmins@uni.lu>

*Have questions or feedback? Your insights matter! Reach out to me at pierrick.pochelu@uni.lu*

# HuggingFace powered with DeepSpeed on HPC


**What is HuggingFace ?**

[HuggingFace](https://huggingface.co/) is a company that provides a variety of open-source Transformers codes, particularly in the domain of Large Language Models (LLM). They are well-known for their huggingface Python package, which offers Transformers architecture codes, and pre-trained models for various tasks. Those codes allow us to build LLM and fine-tune them on custom applications, saving us the need to write complex PyTorch code or train them from scratch.

**What is DeepSpeed ?**

[DeepSpeed](https://www.deepspeed.ai/) is an open-source deep-learning optimization library developed by Microsoft. It is designed to improve the training efficiency and scale of large deep-learning models. DeepSpeed provides a set of capabilities, particularly an optimizer named Zero. Zero optimizer contains:

* Data-parallelism: It splits the batch computing into smaller batches (named local batches), distributing these batches across multiple GPUs, and simultaneously training the model on each local batch to accelerate the training process.
* CPU Offloading: During both the forward and backward steps, it smartly moves the current layer's computations to the GPU for speed, while keeping the other layers stored in the CPU to make use of its larger memory capacity. This allows handle models with billions of parameters that might exceed the GPU memory.
* 16-bit arithmetic for faster computing, faster gradient communication between GPUs, and lower memory footprint.
* List of other features: https://www.deepspeed.ai/docs/config-json/

This tutorial is structured as follow:

* Installation of Anaconda
* Installationf of DeepSpeed, HuggingFace
* HuggingFace+DeepSpeed code
* Performance analysis

This tutorial has been evaluated on Iris (ULHPC) and MeluXina (National HPC).

## Installation of Anaconda
Please skip this if you have already Anaconda. Why using anaconda and not pip directly ? Because anaconda comes with `conda` command which makes it much easier the installation and maintenance of libraries such as CUDA, CUDNN, libaio...

We document 2 ways to install it: option a) using module, or option b) installing from source (more advanced).

#### Step 1 - Option A: locate your HPC module:
`module spider conda` allows you to locate and give you the command to load miniconda.

Example on Iris:

`module load lang/Anaconda3/2020.11`


### Step 1 - Option B: install it yourself


Installing it yourself allows to have a full control of your miniconda installation but note Anaconda may consume more than 35Go. 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Then you can run the script:
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```




### Step 2: Source your environment

**Please don't forget to update the path, example:**
```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/apps/resif/iris-rhel8/2020b/skylake/software/Anaconda3/2020.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/apps/resif/iris-rhel8/2020b/skylake/software/Anaconda3/2020.11/etc/profile.d/conda.sh" ]; then
        . "/opt/apps/resif/iris-rhel8/2020b/skylake/software/Anaconda3/2020.11/etc/profile.d/conda.sh"
    else
        export PATH="/opt/apps/resif/iris-rhel8/2020b/skylake/software/Anaconda3/2020.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```


After this command your terminal invit should look like this:

`(base) 0 [ppochelu@iris-169 ~](3291585 1N/T/1CN)$`

Please note the **(base)** which confirm the success of the previous command block

### Step 3: Create the virtual environment:
Don't forget to do an anaconda virtual environment because you have not the right to update the main environment that the HPC administrators give you.

`conda  create  -n  myenv`

### Step 4: Activate your virtualenvironment
Don't forget to activate your environment each time you will open a new terminal
`conda activate myenv`

you should have something like this:

`(myenv) 0 [ppochelu@iris-169 ~](3291585 1N/T/1CN)$`



## DeepSpeed installation
DeepSpeed is not mandatory if you want to do LLM but allows you to exploit fully HPC witch is required for *really Large* Language Model.

### DeepSpeed installation
Notice: we force the version number for reproducibility purpose. Please feel free to remove the version number (e.g., "==0.12.3") for installing the last version.

```bash
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit # cuda 12.1.0 is compatible with PyTorch 2.1
conda install cudnn==8.9.2.26
pip install torch==2.1.1
pip install torchvision

conda install libaio==0.3.113 # libaio enable disk offloading technics ("NVMe")

DS_BUILD_AIO=1   pip install --no-cache-dir --force-reinstall  deepspeed==0.12.3
```
### Quick DeepSpeed installation check
Installating DeepSpeed enable the command `ds_report`

```
(myenv) 0 [ppochelu@iris-169 ~](3291585 1N/T/1CN)$ ds_report
[2023-11-29 11:43:59,123] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
async_io ............... [YES] ...... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
cpu_adam ............... [NO] ....... [OKAY]
cpu_adagrad ............ [NO] ....... [OKAY]
cpu_lion ............... [NO] ....... [OKAY]
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
evoformer_attn ......... [NO] ....... [NO]
fused_lamb ............. [NO] ....... [OKAY]
fused_lion ............. [NO] ....... [OKAY]
inference_core_ops ..... [NO] ....... [OKAY]
cutlass_ops ............ [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
ragged_device_ops ...... [NO] ....... [OKAY]
ragged_ops ............. [NO] ....... [OKAY]
random_ltd ............. [NO] ....... [OKAY]
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.1
 [WARNING]  using untested triton version (2.1.0), only 1.0.0 is known to be compatible
sparse_attn ............ [NO] ....... [NO]
spatial_inference ...... [NO] ....... [OKAY]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/home/users/ppochelu/.local/lib/python3.8/site-packages/torch']
torch version .................... 2.1.1+cu121
deepspeed install path ........... ['/home/users/ppochelu/.local/lib/python3.8/site-packages/deepspeed']
deepspeed info ................... 0.12.3, unknown, unknown
torch cuda version ............... 12.1
torch hip version ................ None
nvcc version ..................... 12.1
deepspeed wheel compiled w. ...... torch 2.0, cuda 11.7
shared memory (/dev/shm) size .... 377.27 GB
```

3 intersting lines are:
```
torch version .................... 2.1.1+cu121
```
CUDA is correctly seen with 12.1 version


```
nvcc version ..................... 12.1
```
NVCC allows Collective Communication primitive such as Ring All reduce.
```
async_io ............... [YES]
```
which means Non Volatile Memory tensor compute capability are enabled.

### Installing HuggingFace

pip install huggingface[deepspeed]


### PDSH

There command you can do from on MeluXina HPC, but could works on iris too.
If you iris, you can skip because it already contains PDSH from computing nodes.

#### Installing PDSH

```bash
PDSH_ROOT=${PWD} # <---- WARNING replace with the path where do you it is installed
mkdir -p ${PDSH_ROOT}/download
mkdir -p ${PDSH_ROOT}/install
mkdir -p ${PDSH_ROOT}/build

# Download
cd ${PDSH_ROOT}/download
wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/pdsh/source-archive.zip

# Extract the build
unzip source-archive.zip
mv ${PDSH_ROOT}/download/pdsh/ ${PDSH_ROOT}/build/pdsh

cd ${PDSH_ROOT}/build/pdsh/

# Install
./configure --prefix=${PDSH_ROOT}/install --enable-static-modules
make
make install
```
#### Sourcing PDSH

Don't forget to source PDSH everytime you will use DeepSpeed
```bash
LD_LIBRARY_PATH=${PDSH_ROOT}/install/lib/:$LD_LIBRARY_PATH
PATH=${PDSH_ROOT}/install/bin/:$PATH
export PDSH_RCMD_TYPE=ssh
```


## HuggingFace+DeepSpeed code

First, we present a standard complete workflow for a Large Language Model (LLM) using HuggingFace, covering data loading, model training, and predictions. Next, we will modify this code to integrate HuggingFace with DeepSpeed, maximizing the utilization of High-Performance Computing (HPC) resources.

### Standard HuggingFace code

The code contains those steps: data loading, tokenization, fine-tuning, evaluation, saving/restoring and inference.

Notice: we use opt-125m LLM, it is small LLM compared to others, making it ideal for the development phase and quick experiments. Afterwards, we could easily replace `"facebook/opt-125m"` string with bigger and more relevant LLM. The list of huggingFace LLM is foundable there: https://huggingface.co/models

```python
# URL : HuggingFace code samples: https://huggingface.co/docs/transformers/training

model_name="facebook/opt-125m" # <--- select your LLM model

#########################################
# DETERMINISM FOR COMPARING CONVERGENCE #
#########################################
from transformers import enable_full_determinism
enable_full_determinism(42)

#################################
# DATA LOADING (takes 1 minute) #
#################################
from datasets import load_dataset
dataset = load_dataset("yelp_review_full")
print(dataset["train"][10])
print(dataset.shape)

###################################
# DATA PROCESSING (TOKENIZATION)  #
##################################
# Remove the 'label' feature from the dataset because we are not doing classification
dataset = dataset.remove_columns('label')

# ## Tokenize dataset according to pre-trained model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)
n_rows = 1024 # Training on 1024 takes 5 minutes on 1GPU
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(n_rows))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(n_rows))

# ## Data collating
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#################
# MODEL LOADING #
#################
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name) # OPT Model contains 125m parameters

################################
# MODEL TRAINING CONFIGURATION #
###############################
# ## Training hyperparameters
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="test_trainer-125m",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01)

############
# TRAINING #
############
from transformers import Trainer
from accelerate import Accelerator
accelerator = Accelerator() # https://huggingface.co/docs/accelerate/package_reference/accelerator
model = accelerator.prepare(model) # put the model on the GPU
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator) # Training config
trainer.train() # Launch the taining loop
accelerator.wait_for_everyone() # Cleanup accelerator resources after training

########################
# AUTOMATIC EVALUATION #
########################
eval_results = trainer.evaluate()
print(f"Loss: {eval_results['eval_loss']:.2f}")

#########################
# INFERENCE PREPARATION #
########################
trainer.save_model("./opt_trained_model/") # Save the trained model with parameters
del trainer # <--- remove from memory for emulating inference deployment  after training
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./opt_trained_model/")

#############
# INFERENCE #
#############
from transformers import pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=True, device=0)
prompt = "I really love computers because"
pred=generator(prompt)
print(pred)
```

## Code and Scripts
Python code (LLM.py):
The new block of code are highlight with arrows "<----- DeepSpeed"
```bash
import os

# PROCESSES CONFIG <-------------------
pconfig=dict()
pconfig["master_addr"] = os.getenv("MASTER_ADDR", "localhost")
pconfig["master_port"] = int(os.getenv("MASTER_PORT", 9994))
pconfig["rank"] = int(os.getenv("RANK", "0"))
pconfig["local_rank"] = int(os.getenv("LOCAL_RANK", "0"))
pconfig["world_size"] = int(os.getenv("WORLD_SIZE", "1"))
print(pconfig)

# DETERMINISM FOR COMPARING CONVERGENCE
from transformers import enable_full_determinism
enable_full_determinism(42)

# DATA LOADING (takes 1 minute)
from datasets import load_dataset
dataset = load_dataset("yelp_review_full")
print(dataset["train"][10])
print(dataset.shape)


# DATA PROCESSING (TOKENIZATION) 
# Remove the 'label' feature from the dataset because we are not doing classification
dataset = dataset.remove_columns('label')

# ## Tokenize dataset according to pre-trained model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", model_max_length=512)
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)
n_rows = 1024
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(n_rows))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(n_rows))

# ## Data collating
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# MODEL LOADING
# ## Load OPT Model of 125m parameters
model_name="facebook/opt-125m"
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name)

# MODEL TRAINING CONFIGURATION
bs=16
lbs=bs//pconfig["world_size"]
ds_config={
    "per_device_train_batch_size":lbs,
    "train_batch_size": bs,
    "train_micro_batch_size_per_gpu": lbs,
    "optimizer": {"type": "Adam"},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True
    }
}

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="test_trainer-125m",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    per_device_train_batch_size=lbs,
    deepspeed=ds_config # <-------------- activate DeepSpeed
)


# TRAINING
from transformers import Trainer
#from accelerate import Accelerator # <-- Disable accelerator with DeepSpeed
#accelerator = Accelerator()
#model = accelerator.prepare(model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
)

trainer.train()

# Cleanup accelerator resources
#accelerator.wait_for_everyone()

# TESTING
print(model.eval())
eval_results = trainer.evaluate()
print(f"Loss: {eval_results['eval_loss']:.2f}")
```

SLURM launch script (launch_slurm_llm.sh):
```bash
#!/bin/sh -l
#SBATCH -A <your project number>
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH -c 128
#SBATCH -t 40
#SBATCH -N 2
#SBATCH --export=ALL

# get host name
hosts_file="hosts.txt"
scontrol show hostname $SLURM_JOB_NODELIST > $hosts_file

# Collect public key and accept them
while read -r node; do
    ssh-keyscan "$node" >> ~/.ssh/known_hosts
done < "$hosts_file"

# Create the host file containing node names and the number of GPUs
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0;
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

# Don't forget to source PDSH each time you need DeepSpeed
source /project/home/p200117/DistributedDeepLearning/soft/pdsh/env.sh

# Launch HuggingFace+DeepSpeed code by varying the number of GPUs and nodes
deepspeed --num_gpus 1 --num_nodes 1  --hostfile hostfile ./LLM2.py > 1.txt
deepspeed --num_gpus 2 --num_nodes 1  --hostfile hostfile ./LLM2.py > 2.txt
deepspeed --num_gpus 4 --num_nodes 1  --hostfile hostfile ./LLM2.py > 4.txt

# 2 nodes, and 4 GPUs per node, it makes 8 GPUs in total:
deepspeed --num_gpus 4  --num_nodes 2 --hostfile hostfile ./LLM2.py > 8.txt
```

Code launch:
```bash
sbatch ./launch_slurm_llm.sh
```

Results with a combination of `cat` and `grep`:
```
(base) [u101013@login03 deepspeed]$ cat 2.txt | grep 'train_runtime'
{'train_runtime': 190.3416, 'train_samples_per_second': 16.139, 'train_steps_per_second': 1.009, 'train_loss': 2.1075216929117837, 'epoch': 3.0}
(base) [u101013@login03 deepspeed]$ cat 2.txt | grep 'eval_loss'
{'eval_loss': 2.10894775390625, 'eval_runtime': 17.0182, 'eval_samples_per_second': 60.171, 'eval_steps_per_second': 3.761, 'epoch': 1.0}
{'eval_loss': 1.9250223636627197, 'eval_runtime': 12.3188, 'eval_samples_per_second': 83.125, 'eval_steps_per_second': 5.195, 'epoch': 2.0}
{'eval_loss': 1.865960955619812, 'eval_runtime': 12.254, 'eval_samples_per_second': 83.564, 'eval_steps_per_second': 5.223, 'epoch': 3.0}
```

## Performance analysis

| # GPUs | Training time (sec.) | Loss epoch#1  | Loss epoch#2 |  Loss epoch#3 |
| --- | --- |---|---|---|
|      1 |               307 |    1,74 |    1,48 |     1,4 |
|      2 |               190 |    2,11 |    1,93 |    1,87 |
|      4 |               141 |       2 |    1,81 |    1,72 |
|      8 |               116|    1,85 |    1,64 |    1,48 |
|     16 |                114 |       2 |    1,85 |    1,79 |

Conclusion:

* More GPUs decrease the number of computing time.
* The loss changes when we use a different number of GPUs because the order of images isn't the same based on the number of GPUs during training.
* We fixed arbitrary values, more in-depth analysis is required for maximizing convergence speed: learning rate scheduling, batch size, model size.


## Billions parameters LLM


LLM consumes a lot of memory: model's parameters, features flowing through layers, optimizer information, ...  This is why some state-of-art LLM model does not fit entirely in the most advanced GPU memory.

For enabling training model with billions of parameter the HuggingFace+DeepSpeed code, we perform 2 improvements:

* Fix the batch_size per device to the minimum value which is 1 to minimize memory consumption.
* Enable DeepSpeed CPU offloading technics with `"offload_optimizer" and ``"offload_param"`.  

```bash
model_name="facebook/opt-125m"
[...]
lbs=1
bs=lbs*pconfig["world_size"]
ds_config={
  "per_device_train_batch_size":lbs,
  "train_batch_size": bs,
  "train_micro_batch_size_per_gpu": lbs,
  "optimizer": {"type": "AdamW"},
        "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_bucket_size": 1e7,
        "offload_optimizer": {
            "device": "cpu"
         },
        "offload_param": {
            "device": "cpu"
       }
   }
}
[...]
```

Notice: 3D parallelism does not split the memory consumption inside a layers  but spread layers on GPU and CPU. This is why, when we have too big layer it can still crash memory.

## LLM with restricted access
Some HuggingFace LLM requires extra permission such as Meta's Llama2 LLM named `meta-llama/Llama-2-7b-hf`. To use them, you need to follow those steps:

* 1 Create an account on https://huggingface.co/
* Ask permission for a given LLM (example: https://huggingface.co/meta-llama/Llama-2-7b-hf ) . You will receive notification from 24h to 48h.
* Generate a "read" token: https://huggingface.co/settings/tokens and copy the token.
* Call the command: `huggingface-cli login --token <your_token_pasted>`

Change the Python code for calling the model you want. Example:
```
model_name="meta-llama/Llama-2-7b-hf"
[...]
```

