# MNIST Machine Learning (ML) and Deep ML using Tensorflow

See https://ulhpc-tutorials.readthedocs.io/en/latest/deep_learning

You can run the jupyter notebook on the [UL HPC](https://hpc.uni.lu) platform as follows:

```bash
$> ssh -Y -D 1080 iris-cluster
$> ./srun.x11/srun.x11 -p interactive --exclusive -c 28
$> cd ~/tutorials/ML-DL/
$> module load lang/Python/2.7.13-foss-2017a
$> source dlenv/bin/activate
$> jupyter notebook password
$> jupyter notebook --ip $(ip addr show em1 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser
```
