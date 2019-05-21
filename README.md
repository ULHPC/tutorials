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
   You need to specify your installation destination, e.g. `/home/users/<your username>/tools/miniconda3`. You must use the **full** path and can**not** user `$HOME/tools/miniconda3`. Answer `yes` to initialize Miniconda3. 

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
   You can see that your prompt will now be prefixed with `(bioinfo_tutorial)` to show which environment is active. For the rest of the tutorial make sure that you always have this environment active.
4. Install snakemake:

   ```bash
   (node)$> conda install -c bioconda -c conda-forge snakemake-minimal
   ```
   
   

## Create snakemake workflow

In this tutorial we will analyse ChIP-seq data [3] from a paper published by our colleagues in LSRU [4].

To speed up computing time we use source files that only contain sequencing reads that map to chromosome 12. The files for input (control) and H3K4me3 (ChIP) are available on the cluster in the directory `/work/projects/ulhpc-tutorials/bio/snakemake/chip-seq` and the corresponding reference in `/work/projects/ulhpc-tutorials/bio/snakemake/reference`.

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
  input: "chip-seq/H3K4-TC1-ST2-D0.12.fastq.gz "
  output: "bowtie2/H3K4-TC1-ST2-D0.12.bam"
  shell:
    """
    bowtie2 -x reference/Mus_musculus.GRCm38.dna_sm.chromosome.12 -U {input} | \  
    samtools sort - > {output}
    samtools index {output}
    """
```

Since we have two fastq files to map, we should generalise the rule with wildcards:

```python
rule mapping:
  input: "chip-seq/{sample}.fastq.gz"
  output: "bowtie2/{sample}.bam"
  shell:
    """
    bowtie2 -x reference/Mus_musculus.GRCm38.dna_sm.chromosome.12 -U {input} | \  
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

We also use the `params` directive to define the path to the reference and declutter the command-line call, as well as the `log` directive to define a path to permanently store the output of the execution. This is especially useful in this step to store the bowtie2 mapping statistics, which are just written to the command-line (stderr) otherwise.

To track resource usage we add the `benchmark` directive, which will write performance measures to a tsv file.

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
    idx = "reference/Mus_musculus.GRCm38.dna_sm.chromosome.12"
  log: "logs/bowtie2_{sample}.log"
  benchmark: "benchmarks/mapping/{sample}.tsv"
  conda: "envs/bowtie2.yaml"
  shell:
    """
    bowtie2 -x {params.idx} -U {input} 2> {log} | \
    samtools sort - > {output}
    samtools index {output}
    """
```

You can test the rule by specifying one of the potential outputs. We first do a dry-run with with option `-n`.

```bash
(node)$> snakemake -npr --use-conda bowtie2/INPUT-TC1-ST2-D0.12.bam
```

If everything is fine we can run the rule to create the file `bowtie2/INPUT-TC1-ST2-D0.12.bam`:

```bash
(node)$> snakemake -pr --use-conda bowtie2/INPUT-TC1-ST2-D0.12.bam
```

Check the benchmark report:

```bash
(node)$> cat benchmarks/mapping/INPUT-TC1-ST2-D0.12.tsv
s       h:m:s   max_rss max_vms max_uss max_pss io_in io_out mean_load
19.1737 0:00:19 262.14  1404.55 258.79  258.94  0.00  0.00   0.00
```



### Peak calling

The next step in the workflow is to call peaks with MACS2 [9]. This tells us where there is enrichment of the ChIP versus the input (control). 

You should always choose the peak caller based on how you expect your enriched regions to look like, e.g. narrow or broad peaks.

Besides the list of peaks in BED format, MACS2 also produces coverage tracks.

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

The conda environment `envs/macs2.yaml` for this step is:

```yaml
name: macs2
channels:
  - bioconda
dependencies:
  - macs2
```

Let's run this step with:

```bash
(node)$> snakemake -pr --use-conda output/TC1-ST2-D0.12_peaks.narrowPeak
```

Note that snakemake will not run the mapping step for `bowtie2/INPUT-TC1-ST2-D0.12.bam` again. It only runs rules for which the output is not present or the input has changed.

### Generate bigwig files for visualisation

For easier visualisation and faster transfer, we convert the two coverage tracks from the MACS2 output to bigWiggle format.

```python
rule bigwig:
  input: "macs2/{sample}.bdg"
  output: "output/{sample}.bigwig"
  params:
    idx = "reference/Mus_musculus.GRCm38.dna_sm.chromosome.12.fa.fai"
  conda: "envs/ucsc.yaml"
  shell:
    """
    bedGraphToBigWig {input} {params.idx} {output}
    """
```

The conda environment `envs/ucsc.yaml` for this step is:

```yaml
name: ucsc
channels:
  - bioconda
dependencies:
  - ucsc-bedgraphtobigwig
```

Let's test this step with:

```bash
(node)$> snakemake -pr --use-conda output/TC1-ST2-D0.12_treat_pileup.bigwig
```

This time snakemake will only run the "bigwig" rule for the one file we specified.



### Summary rule

To avoid always having to specify which output file we want on the command-line, we add one rule with just inputs that defines the result files we want to have in the end. Since by default snakemake executes the first rule in the snakefile, we need add this rule as the first one to the top, and then we don't need to specify anything additional on the command-line.

First, at the **very top** of the Snakefile, define a variable for the name of the sample:

```python
SAMPLE = "TC1-ST2-D0.12"
```

This makes it easier to change the Snakefile and apply it to other datasets. Snakemake is based on Python so we can use Python code inside the Snakefile. We will use f-Strings [10] to include the variable in the file names.

Add this rule at the top of the `Snakefile` after the line above:

```python
rule all:
  input: f"output/{SAMPLE}_peaks.narrowPeak", f"output/{SAMPLE}_control_lambda.bigwig", f"output/{SAMPLE}_treat_pileup.bigwig"
```

Finally run the workflow again, this time without a specific target file:

```bash
(node)$> snakemake --use-conda -pr
```

Snakemake can visualise the dependency graph of the workflow with the following command:

```bash
(node)$> snakemake --dag | dot -Tpng > dag.png
```

![DAG](dag.png)

## Cluster configuration for snakemake

Until now the workflow just runs on a single CPU on a single machine, which is not very efficient when we have much more resources available. To speed up the computation you should check in the documentation of the software you use how it can scale. For bioinformatics tools the most common option is multithreading.

In this workflow only bowtie2 has the option to run on multiple threads.

### Adjust mapping step to run on multiple threads

We add the `thread` directive to the snakemake rule for the mapping step, to tell snakemake that this step can use multiple threads. We also need to add the option `-p` to the bowtie2 command-line call, to make it actually use those threads.

```python
rule mapping:
  input: "chip-seq/{sample}.fastq.gz"
  output: "bowtie2/{sample}.bam"
  params:
    idx = "reference/Mus_musculus.GRCm38.dna_sm.chromosome.12"
  log: "logs/bowtie2_{sample}.log"
  benchmark: "benchmarks/mapping/{sample}.tsv"
  conda: "envs/bowtie2.yaml"
  threads: 4
  shell:
    """
    bowtie2 -p {threads} -x {params.idx} -U {input} 2> {log} | \
    samtools sort - > {output}
    samtools index {output}
    """
```

TODO: new slurm job with more cores; remove results with clean rule

Now we also need to tell snakemake that it has multiple cores available and can run steps multithreaded or run multiple tasks in parallel. This is done with `-j` option followed by the number of available cores (e.g. the number of cores you have reserved if you run it interactively).

```bash
snakemake -j 4 -npr --use-conda bowtie/TC1-I-ST2-D0.12.bam
```

You should see in the output that the command-line call of bowtie2 now shows `-p 4`.

TODO:  test several options for -j and compare benchmark



### Configure job parameters with `cluster.json`

Instead of reserving an interactive job and running snakemake inside that job, we want to use snakemake's cluster functionality to make it submit jobs to SLURM. For this we create configuration file named `cluster.json` to define the values for the different `sbatch` options.

Options under the `__default__` header apply to all rules, but it's possible to override them selectively with rule-specific options.

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

The meaning of the option `-j` changes when running in cluster mode to denote the maximum number of simultaneous jobs.

In order to make snakemake run all the steps again, we need to the delete the output folders.

```bash
(access)$> cd $SCRATCH/bioinfo_tutorial
(access)$> rm -rf bowtie2/ macs2/ output/ # please be careful with this command!
(access)$> conda activate bioinfo_tutorial
 
(access)$> SLURM_ARGS="-p {cluster.partition} -N 1 -n {cluster.n} -c {cluster.ncpus} -t {cluster.time} --job-name={cluster.job-name} -o {cluster.output} -e {cluster.error}"
 
(access)$> snakemake -j 10 -pr --use-conda --cluster-config cluster.json --cluster "sbatch $SLURM_ARGS"
```

 Check again the benchmark report:

```bash
(access)$> cat benchmarks/mapping/INPUT-TC1-ST2-D0.12.tsv
s      h:m:s   max_rss max_vms max_uss max_pss io_in io_out mean_load
6.7687 0:00:06 295.01  1728.68 291.64  291.79  0.00  16.00  0.00
```

Notice that the runtime has decreased, but I/O has increased.



## Inspect results in IGV

Now that we have completed the workflow, let's have a look at the results.

For visualisation, [download IGV](http://software.broadinstitute.org/software/igv/download), or use any other genome browser of your choice.

To copy the results from the cluster to your laptop, run the following and replace `<your username>` with your ULHPC user login. Pay attention in which directory you are, so you can find the files again.

```bash
(laptop)$> mkdir bioinfo_tutorial
(laptop)$> cd bioinfo_tutorial
(laptop)$> rsync -avz iris-cluster:/scratch/users/<your username>/bioinfo_tutorial/output .
```

Start IGV and select mouse mm10 as genome in the drop-down menu in the upper left. Go to "File" -> "Load from File…" and select all three files that you have copied from the cluster.

In the search box enter for example "Ahr" to check the signal around one of the genes highlighted in the paper.

Pay attention to the scale on which the two coverage tracks are shown. You can right-click on the track name on the left and select "Autoscale" to adjust the range.

TODO: screenshot of IGV



## (optional) Plot enrichment with deepTools



## Useful stuff

* To avoid too much overhead in the number of jobs submitted to SLURM, use the`group` directive to group rules that can run together in a single job.



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
