# Bioinformatics workflows with snakemake and conda

Author: Sarah Peter

In this tutorial you will learn how to run a ChIP-seq analysis with the snakemake workflow engine [1] on the cluster.

**Disclaimer:** In order to keep this tutorial simple, we use default parameters for the different tools as much as possible. However, for a real analysis you should always adapt the parameters to your dataset. Also, be aware that the results of some steps might be screwed, because we only work on data from one chromosome.

## Setup the environment

For this tutorial we will use the `conda` [2] package manager to install the required tools. It can encapsulate software and packages in environments, so you can have multiple different versions of a software installed at the same time and avoid incompatibilities between different tools. It also has functionality to easily port and replicate environments, which is important to ensure reproducibility of analyses.

We will use conda on two levels in this tutorial. First we use a conda environment to install and run snakemake. Second, inside the snakemake workflow we will define separate conda environments for each step.

1. Start a job on the cluster:

   ```bash
   (laptop)$> ssh iris-cluster
   (access)$> si
   ```

2. Install conda:

   ```bash
   (node)$> mkdir -p $SCRATCH/downloads
   (node)$> cd $SCRATCH/downloads
   (node)$> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   (node)$> chmod u+x Miniconda3-latest-Linux-x86_64.sh
   (node)$> ./Miniconda3-latest-Linux-x86_64.sh 
   ```
   You will need to specify your installation destination, e.g. `/home/users/sdiehl/tools/miniconda3`. You must use the **full** path and can**not** user `$HOME/tools/miniconda3`. Answer `yes` to initialize Miniconda3. 

   The installation will modify your `.bashrc` to make conda directly available after each login. To activate the changes now, run

   ```bash
   (node)$> source ~/.bashrc
   ```

   Update conda to the latest version:
   ```bash
   (node)$> conda update conda 
   ```

3. Create a new conda environment and activate it:

   ```bash
   (node)$> conda create -n bioinfo_tutorial
   (node)$> conda activate bioinfo_tutorial
   ```
   You can see that your prompt will now be prefixed with `(bioinfo_tutorial)` to show which environment is active. For rest of the tutorial make sure that you always have this environment active.
4. Install snakemake:

   ```bash
   (node)$> conda install -c bioconda -c conda-forge snakemake-minimal
   ```
   
   

## Create snakemake workflow

In this tutorial we will analyse ChIP-seq data [3] from a paper recently published by our colleagues in LSRU [4].

To speed up computing time we will use source files that only contain sequencing reads from chromosome 7. The files for input (control) and H3K4me3 (ChIP) are available on the cluster in the directory `/work/projects/ulhpc-tutorials/bio/snakemake/chip-seq` and the corresponding reference in `/work/projects/ulhpc-tutorials/bio/snakemake/reference`.

We have also already prepared the conda environments for each step in the workflow in `/work/projects/ulhpc-tutorials/bio/snakemake/envs`.

Create a working directory and link the necessary data:

```bash
(node)$> cd $SCRATCH
(node)$> mkdir bioinfo_tutorial
(node)$> cd bioinfo_tutorial
(node)$> ln -s /work/projects/ulhpc-tutorials/bio/snakemake/chip-seq .
(node)$> ln -s /work/projects/ulhpc-tutorials/bio/snakemake/reference .
(node)$> ln -s /work/projects/ulhpc-tutorials/bio/snakemake/envs .
```

### Mapping

"In Snakemake, workflows are specified as Snakefiles. Inspired by GNU Make, a Snakefile contains rules that denote how to create output files from input files. Dependencies between rules are handled implicitly, by matching filenames of input files against output files. Thereby wildcards can be used to write general rules." [5]

"Most importantly, a rule can consist of a name (the name is optional and can be left out, creating an anonymous rule), input files, output files, and a shell command to generate the output from the input, i.e." [6]

```python
rule NAME:
    input: "path/to/inputfile", "path/to/other/inputfile"
    output: "path/to/outputfile", "path/to/another/outputfile"
    shell: "somecommand {input} {output}"
```



A basic rule for mapping a fastq file with bowtie2 [7] could look like this:

```python
rule mapping:
  input: "chip-seq/TC1-H3K4-ST2-D0.7.fastq.gz"
  output: "bowtie/TC1-H3K4-ST2-D0.7.bam"
  shell:
    """
    bowtie2 -x reference/Mus_musculus.GRCm38.dna_sm.chromosome.7 -U {input} | \  
    samtools sort - > {output}
    samtools index {output}
    """
```

Since we have two fastq files to map, we should generalise the rule with wildcards:

```python
rule mapping:
  input: "chip-seq/{sample}.fastq.gz"
  output: "bowtie/{sample}.bam"
  shell:
    """
    bowtie2 -x reference/Mus_musculus.GRCm38.dna_sm.chromosome.7 -U {input} | \  
    samtools sort - > {output}
    samtools index {output}
    """
```

Now we need to tell snakemake to use a conda environment with bowtie2 and samtools [8] inside to run this rule. For this purpose there is a specific `conda` directive that can be added to the rule. It accepts a yaml file that defines the conda environment. 

```python
conda: "envs/bowtie2.yaml"
```

You can easily export existing conda environments to a yaml file with `conda env export` or write the yaml from scratch. For this step the yaml file looks like this:

```yaml
name: bowtie2
channels:
  - bioconda
dependencies:
  - bowtie2
  - samtools
```

We will also use the `params` directive to define the path to the reference and declutter the command-line call, as well as the `log` directive to define a path to permanently store the output of the execution. This is especially useful in this step to store the bowtie2 mapping statistics, which are just written to the command-line (stderr) otherwise.

Create a file called `Snakefile` in the current directory and open it in your favourite editor, e.g.

```bash
(node)$> nano Snakefile
```

Add the final rule for the mapping:

```python
rule mapping:
  input: "chip-seq/{sample}.fastq.gz"
  output: "bowtie2/{sample}.bam"
  params:
    idx = "reference/Mus_musculus.GRCm38.dna_sm.chromosome.7"
  log: "logs/bowtie2_{sample}.log"
  conda: "envs/bowtie2.yaml"
  shell:
    """
    bowtie2 -x {params.idx} -U {input} 2> {log} | \
    samtools sort - > {output}
    samtools index {output}
    """
```

You can test the rule by specifying one of the potential outputs. We will just do a dry-run with with option `-n` for now.

```bash
(node)$> snakemake -npr --use-conda bowtie2/TC1-I-ST2-D0.7.bam
```



### Peak calling

Macs2 [9]

```python
rule peak_calling:
  input:
    control = "bowtie2/INPUT-{sample}.bam",
    chip = "bowtie2/H3K4-{sample}.bam"
  output:
    peaks = "output/{sample}_peaks.narrowPeak",
    control_bdg = "macs2/{sample}_control_lambda.bdg",
    chip_bdg = "macs2/{sample}_treat_pileup.bdg"
  conda: "envs/macs2.yaml"
  shell:
    """
    macs2 callpeak -t {input.chip} -c {input.control} -f BAM -g mm -n {wildcards.sample} -B -q 0.01 --outdir macs2

    cp macs2/{wildcards.sample}_peaks.narrowPeak {output.peaks}
    """
```

### Generate bigwig files for visualisation

```python
rule bigwig:
  input: "macs2/{sample}.bdg"
  output: "output/{sample}.bigwig"
  params:
    idx = "reference/Mus_musculus.GRCm38.dna_sm.chromosome.7.fa.fai"
  conda: "envs/ucsc.yaml"
  shell:
    """
    bedGraphToBigWig {input} {params.idx} {output}
    """
```



### Summary rule

At the **very top** of the Snakefile, define a variable for the name of the sample:

```python
SAMPLE = "TC1-ST2-D0.7"
```

This makes it easier to change the Snakefile and apply it to other datasets. Snakemake is based on Python so we can use Python code inside the Snakefile. We will use f-Strings [10] to include the variable in the file names.

Add this rule at the top of the `Snakefile` after the line above:

```python
rule all:
  input: f"output/{SAMPLE}_peaks.narrowPeak", f"output/{SAMPLE}_control_lambda.bigwig", f"output/{SAMPLE}_treat_pileup.bigwig"
	
```

Finally test the workflow again, this time without a specific target file:

```bash
(node)$> snakemake --use-conda -npr
```

Snakemake can visualise the dependency graph of the workflow with the following command:

```bash
(node)$> snakemake --dag | dot -Tpng > dag.png
```

![DAG](dag.png)

## Cluster configuration for snakemake

### Adjust mapping step to run on multiple threads

```python
rule mapping:
  input: "chip-seq/{sample}.fastq.gz"
  output: "bowtie2/{sample}.bam"
  params:
    idx = "reference/Mus_musculus.GRCm38.dna_sm.chromosome.7"
  log: "logs/bowtie2_{sample}.log"
  conda: "envs/bowtie2.yaml"
  threads: 4
  shell:
    """
    bowtie2 -p {threads} -x {params.idx} -U {input} 2> {log} | \
    samtools sort - > {output}
    samtools index {output}
    """
```

Run test:

We also need to tell snakemake to run multithreaded or run multiple tasks in parallel with `-j` option.

```bash
snakemake -j 4 -npr --use-conda bowtie/TC1-I-ST2-D0.7.bam
```



### Configure job parameters with `cluster.json`

```json
{
    "__default__" :
    {
        "time" : "0-00:10:00",
        "n" : 1,
        "partition" : "batch",
        "ncpus": 1,
        "job-name" : "{rule}",
        "output" : "slurm-%j-%x.out",
        "error" : "slurm-%j-%x.err"
    },
    "mapping":
    {
        "ncpus": 4,
    },
}
```



### Run snakemake with cluster configuration

Make sure you quit your job and run the following from the access node. Now we need to map the variables defined in `cluster.json` to the command-line parameters of `sbatch`.

```bash
 (access)$> conda activate bioinfo_tutorial
 
 (access)$> SLURM_ARGS="-p {cluster.partition} -N 1 -n {cluster.n} -c {cluster.ncpus} -t {cluster.time} --job-name={cluster.job-name} -o {cluster.output} -e {cluster.error}"
 
 (access)$> snakemake -j 10 -pr --use-conda --cluster-config cluster.json --cluster "sbatch $SLURM_ARGS"
```

 

## Inspect results in IGV

Download IGV from http://software.broadinstitute.org/software/igv/download.

```
rsync -avz iris-cluster:/scratch/users/sdiehl/bioinfo_tutorial/output .
```

Select mouse mm10 as genome in the upper left.

Use `TC1-ST2-H3K4-D0_peaks.narrowPeak` to visualize peaks.



## (optional) Plot enrichment with deepTools





## References

* [1] [Snakemake](https://snakemake.readthedocs.io/en/stable/)
* [2] [Anaconda](https://www.anaconda.com/)
* [3] [ChIP-seq data](https://www.ebi.ac.uk/ena/data/view/PRJEB20933)
* [4] [Gérard D, Schmidt F, Ginolhac A, Schmitz M, Halder R, Ebert P, Schulz MH, Sauter T, Sinkkonen L. Temporal enhancer profiling of parallel lineages identifies AHR and GLIS1 as regulators of mesenchymal multipotency. *Nucleic Acids Research*, Volume 47, Issue 3, 20 February 2019, Pages 1141–1163, https://doi.org/10.1093/nar/gky1240](https://www.ncbi.nlm.nih.gov/pubmed/30544251)
* [5] https://snakemake.readthedocs.io/en/stable/snakefiles/writing_snakefiles.html
* [6] https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html
* [7] [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml)
* [8] [Samtools](http://www.htslib.org/)
* [9] [MACS2](https://github.com/taoliu/MACS)
* [10] [Python f-Strings](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings)
* deepTools

## Acknowledgements

Many thanks to Nikola de Lange, Aurélien Ginolhac, Roland Krause and Cedric Laczny for their help in developing this tutorial.