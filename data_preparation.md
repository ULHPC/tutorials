# Data preparation

This document describes how to prepare the data for the tutorial.



## Download

ChIP-seq data:

* H3K4me3: https://www.ebi.ac.uk/ena/data/view/ERR2008263
* Input: https://www.ebi.ac.uk/ena/data/view/ERR2014243

```bash
wget "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR200/003/ERR2008263/ERR2008263.fastq.gz"
wget "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR201/003/ERR2014243/ERR2014243.fastq.gz"
mv ERR2008263.fastq.gz TC1-H3K4-ST2-D0.fastq.gz
mv ERR2014243.fastq.gz TC1-INPUT-ST2-D0.fastq.gz
```

Reference genome GRCm38:

```bash
wget "ftp://ftp.ensembl.org/pub/release-96/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna_sm.toplevel.fa.gz"
gunzip Mus_musculus.GRCm38.dna_sm.toplevel.fa.gz
ln -s Mus_musculus.GRCm38.dna_sm.toplevel.fa Mus_musculus.GRCm38.dna_sm.toplevel.fasta
```



## BWA index

```bash
srun -p batch --qos qos-batch -t 0-8:00:0 -N 1 -c 1 --ntasks-per-node=14 --pty bash
bwa index Mus_musculus.GRCm38.dna_sm.toplevel.fa
```



## Mapping

Follow the "Setup" and "Mapping" steps from http://ginolhac.github.io/chip-seq, but use full genome as reference.