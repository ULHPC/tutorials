# Introduction to containers on ULHPC

## Advantages of containers

- Portability: run the same experiments on multiple environments (HPC centers, local machine, cloud services) without having to set up all necessary software on each environment.
- Reproducibility: you can re-run the same experiment years later without worrying about:
    - which version of your software or its dependencies were used when you first ran your experiment
    - which operating system version / modules are currently on the HPC 
- Simplify team-work: 
    - Share your container in your team to ensure everyone works with the same software environment
    - Version your containers to use new software version while still keeping older version for reproducibility
    - Onboard new team member faster
- Simplify peer-review: share your container to help other teams anywhere in the world to quickly reproduce your experiments

## Limitations

On the ULHPC, you cannot:

- build or run Docker containers 
- build a Singularity container from a definition file

On the ULHPC, you can:

- build a Singularity container from a Docker container
- run any Singularity container

To build, you can:

- use your own machine
- use our [container Gitlab project](https://gitlab.uni.lu/hlst/ulhpc-containers) and benefit from its CI / container registry

## Pratical sessions plan

1. Convert a publicly available Docker container to a Singularity container
2. Build a custom Docker container and convert it into a Singularity container
3. Use GPUs in a Singularity container

# Practical session 1: Python container

The objective of this section are:

- to create a simple container to be used on the ULHPC
- to learn how to use the container

## Creation of the container

First you need to connect to the ULHPC, preferably on AION.
Then you need to request an interactive session as you cannot perform the following action from an access node.

```Bash
[jschleich@access1 ~]$ si
```

Then, you first need to load the Singularity module to be able to use it.
After that step, you can pull (download) an already available Docker container from [Dockerhub](https://hub.docker.com/) and convert it into a Singularity container.

When the conversion is over, you have one new file which is named after the container and its version.

```Bash
$ module load tools/Singularity
$ singularity pull docker://python:3.10
[...]
$ ll
total 307712
-rwxr-xr-x 1 jschleich clusterusers 314679296 May 30 10:43 python_3.10.0.sif
```

Note that if you prefer another version that `3.10`, you can browse [the available versions](https://hub.docker.com/_/python) and amend the command line.

## How to use the container

This section introduces `run` and `exec` briefly. For more information, the documentation of Singularity / Apptainer (fork of Singularity) can be found [here](https://apptainer.org/user-docs/master/).

### Run vs Exec

The singularity `run` command is used to execute a Singularity container as if it were a standalone program. It runs the container image and executes the default action defined within the container. It is similar to directly running an executable file, but within the isolated environment provided by the Singularity container.

On the other hand, the singularity `exec` command allows you to execute a specific command or script within a Singularity container. It provides a way to run a specific program or script inside the container without executing the default action defined within the container.

In summary, singularity run is used to execute the default action defined in the container, while singularity exec is used to run a specific command or script within the container.

Let's first use `run` on with our new container. Without surprise, a Python prompts is displayed.
From here, you can type Python code which will be run directly in the container.

### Singularity run

```Bash
$ singularity run python_3.10.0.sif
Python 3.10.0 (default, Dec  3 2021, 00:21:30) [GCC 10.2.1 20210110] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### Singularity exec

Let's now try `exec`. As we do not use the default action of the container, we need to specify what should be executed. In this example, I chose `bash` in order to execute a new interactive Bash shell. We are welcomed with a new prompt, `Singularity>` and then we can type in any commmand. Again, these commands will be exectuted from inside the container.

```Bash
$ singularity exec python_3.10.0.sif bash
Singularity> python --version
Python 3.10.10
```

### Binding feature

Now that we know how to run things in a container, we need to be able to interact with outside of the container, e.g., to access input files or to write output files. To do so, we can specify a list of directories (and even files) from outside the container, e.g., in our home / project / scratch directories to be accessible from inside the container.

When you use the `--bind` or `-B` option with Singularity, you specify a binding between a directory (or file) on the host system (in our case the ULHPC cluster) and a directory inside the container. This creates a link between the two, enabling the container to read from or write to the specified host directory or file.

The following example considers a Python script that will write some dummy content to a file in a specific output directory. The script itself is stored in an input directory.

The following folder structure is assumed:
```Console
.
|-- input
|   `-- script.py
|-- ouput
|   `-- my-file
|-- python_3.10.0.sif
```

The Python file `script.py` is as follows:

```Python
import os
import sys

def write_file(folder_path, file_name, content):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as file:
        file.write(content)
    print(f"File '{file_name}' created in '{folder_path}'.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <folder_path> <file_name>")
        sys.exit(1)

    folder_path = sys.argv[1]
    file_name = sys.argv[2]
    content = "Dummy content for the purpose of the demonstration"

    write_file(folder_path, file_name, content)
```

To execute the script in the container, see the following command:

```Bash
singularity exec \
	-B input/:/container-input \
	-B ouput:/container-output \
	python_3.10.0.sif \
	python /container-input/script.py /container-output my-file
```

The command does the following:

- It binds the `input` folder of the host to a container folder `/container-input`
- It binds the `output` folder of the host to a container folder `/container-output`
- It executes the file `script.py` which is located **in** the container at the following address: `/container-input/script.py` 
- It outputs a file named `my-file` to the container folder `/container-output` which is also `output` on the host file system.

Note: you can amend all folder names as you wish and you could use the same host directory for both input and output.

### Auto-binding feature 

By default, Singularity automatically binds several directories and in particular it binds the home folder of the host. This features simplifies the usage of Singularity for the users, however it can also lead to unexpected behaviours and frustration.

This is particularly true if you have different Python packages installed in your host home folder and in the container as the container may use the host packages instead of the container one.

This behaviour can be observed below.
Here we can see all the packages which can be seen from inside the container. The list was truncated for the sake of conciseness.
The container was not built with all those packages, however the container sees them because the auto-binding feature led to host packages being available to the container. 

```Bash
$ singularity run python_3.10.0.sif bash
Singularity> pip list
Package                 Version
----------------------- ---------
boltons                 23.0.0
brotlipy                0.7.0
certifi                 2022.12.7
cffi                    1.15.1
charset-normalizer      2.0.4
conda                   23.3.1
conda-content-trust     0.1.3
conda-package-handling  2.0.2
conda_package_streaming 0.7.0
cryptography            38.0.4
idna                    3.4
jsonpatch               1.32
jsonpointer             2.0
```

We can prevent that behaviour by adding the `--no-home` option and we can see that the list of packages is much smaller as it only contains the one from the container.

```Bash
$ singularity run --no-home python_3.10.0.sif bash
Singularity> pip list
Package    Version
---------- -------
pip        21.2.4
setuptools 57.5.0
wheel      0.37.0
```

Note: you can use the `--home` option to specify a specific host directory to be mounted as the container home folder. See below:

```Bash
$ singularity run --home `pwd` python_3.10.0.sif bash
Singularity> pip list
Package    Version
---------- -------
pip        21.2.4
setuptools 57.5.0
wheel      0.37.0
```

# Practical session 2: custom R container

The objective of this section are:

- to create a simple container to be used on the ULHPC
- to learn how to use the container

## Container specification and build

```Docker
FROM registry.gitlab.uni.lu/hlst/ulhpc-containers/r4:4.2.2
ENV LC_ALL "C"
RUN R --slave -e 'install.packages("ggplot2",repos="https://cran.rstudio.com/")'
```

The above Docker file does the following things:

- It requests a version of R which is maintained by the ULHPC team (r4:4.2.2) and use it as a base
- It sets the LC_ALL environment variable to prevent some compilation issues
- It installs the `ggplot2` package. You can add as many package as you require for your experimental setup.

The build process, including testing of the container image is performed by Gitlab CI (Continuous Integration). 
The built container is then stored in the [Gitlab container registry](https://gitlab.uni.lu/hlst/ulhpc-containers/container_registry).

## Use the container on ULHPC

### Connection to the ULHPC
First you need to connect to the ULHPC, preferably on AION.
Then you need to request an interactive session as you cannot perform the following action from an access node:

```Bash
[jschleich@access1 ~]$ si
```

### Conversion into a Singularity container

Then you need to load the Singularity module and then you can pull the Docker container from the Gitlab container registry and convert it into a Singularity container:

```Bash
$ module load tools/Singularity
$ singularity pull docker://registry.gitlab.uni.lu/hlst/ulhpc-containers/r-tutorial:1.0.0
[...]
$ ll
-rwxr-xr-x   1 jschleich clusterusers 627M May 30 15:35 r-tutorial_1.0.0.sif
```

### Execution of the container

You can then run the container simply by doing:

```Bash
singularity run --no-home r-tutorial_1.0.0.sif
```

You should see the following result:
```Console
R version 4.2.2 (2022-10-31) -- "Innocent and Trusting"
Copyright (C) 2022 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.
```

# Practical session 3: use GPUs with container

The objective of this section is to learn how to use the GPU with Singularity. 

## Pytorch container on GPU node

First you need to book a GPU from a GPU node.
This gives you one GPU, one core for 30 min, which should be more than enough to test the following intructions.

```Bash
si-gpu
```

Then, you need to load the Singularity module:

```Bash
module load tools/singularity
```

You can copy the following script which uses PyTorch and requests a GPU to work.

```Python
import torch
import math

dtype = torch.float
device = torch.device("cuda:0")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

```

On the ULHPC, several Singularity containers are maintained via the [ULHPC container project](https://gitlab.uni.lu/hlst/ulhpc-containers). This is a public project and you are encouraged to use it for you own containers.

Some of those maintained containers are Singularity container and can be found at the following address `/work/projects/singularity/ulhpc/`.
The other containers are Docker containers which are stored and versioned in the [container registry](https://gitlab.uni.lu/hlst/ulhpc-containers/container_registry).

For our practical session, we will use a Singularity container which can be found at: `/work/projects/singularity/ulhpc/pytorch-22.04.sif`.

If you try to execute the container via the following command:
```Bash
singularity run /work/projects/singularity/ulhpc/pytorch-22.04.sif
```

You will notice the following message:

```Console
WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
```

It means that the container does not have access to the GPU of the host node.
If you try to execute the aforementioned Python script, you will see the following error:

```Console
Traceback (most recent call last):
  File "pytorch-test-gpu.py", line 9, in <module>
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
  File "/opt/conda/lib/python3.8/site-packages/torch/cuda/__init__.py", line 216, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```

In order to access the GPU, you have to add the `--nv` option. Under the hood, what that does is that Singularity will mount / bind whatever it needs from the host node to use the NVIDIA GPU.

```Bash
singularity run --nv /work/projects/singularity/ulhpc/pytorch-22.04.sif python pytorch-test-gpu.py
```

## Other containers using GPU

Have a look here: `/work/projects/singularity/ulhpc` and do not hesitate to request some more.

# Final remark

Please use the [ULHPC container project](https://gitlab.uni.lu/hlst/ulhpc-containers) and do not hesitate to contact me in case of doubt on how to use it.

