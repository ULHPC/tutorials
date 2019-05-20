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

Follow the "Setup" and "Mapping" steps from http://ginolhac.github.io/chip-seq, but use the full genome as reference.



## Extracting reads from one chromosome

Extracting the reads:

```bash
samtools index TC1-H3K4-ST2-D0.GRCm38.q30.bam
samtools index TC1-I-ST2-D0.GRCm38.q30.bam
samtools view -h -b TC1-H3K4-ST2-D0.GRCm38.q30.bam 12 > TC1-H3K4-ST2-D0.GRCm38.q30.12.bam
samtools view -h -b TC1-I-ST2-D0.GRCm38.q30.bam 12 > TC1-I-ST2-D0.GRCm38.q30.12.bam
```

Reverting to fastq format:

```
conda install -c bioconda picard

picard RevertSam I=TC1-H3K4-ST2-D0.GRCm38.q30.12.bam O=TC1-H3K4-ST2-D0.GRCm38.q30.12.revertsam.bam SANITIZE=true MAX_DISCARD_FRACTION=0.005 ATTRIBUTE_TO_CLEAR=XT ATTRIBUTE_TO_CLEAR=XN ATTRIBUTE_TO_CLEAR=AS ATTRIBUTE_TO_CLEAR=OC ATTRIBUTE_TO_CLEAR=OP

picard RevertSam I=TC1-I-ST2-D0.GRCm38.q30.12.bam O=TC1-I-ST2-D0.GRCm38.q30.12.revertsam.bam SANITIZE=true MAX_DISCARD_FRACTION=0.005 ATTRIBUTE_TO_CLEAR=XT ATTRIBUTE_TO_CLEAR=XN ATTRIBUTE_TO_CLEAR=AS ATTRIBUTE_TO_CLEAR=OC ATTRIBUTE_TO_CLEAR=OP

samtools fastq TC1-H3K4-ST2-D0.GRCm38.q30.12.revertsam.bam > TC1-H3K4-ST2-D0.12.fastq
samtools fastq TC1-I-ST2-D0.GRCm38.q30.12.revertsam.bam > TC1-I-ST2-D0.12.fastq
gzip TC1-H3K4-ST2-D0.12.fastq
gzip TC1-I-ST2-D0.12.fastq
```



## Creating reference for one chromosome

Extract single chromosome from fasta:

```bash
samtools faidx Mus_musculus.GRCm38.dna_sm.toplevel.fasta 12 > Mus_musculus.GRCm38.dna_sm.chromosome.12.fa
```

or just download the single chromosome fasta from Ensembl:

```bash
wget "ftp://ftp.ensembl.org/pub/release-96/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna_sm.chromosome.12.fa.gz"
gunzip Mus_musculus.GRCm38.dna_sm.chromosome.12.fa.gz
```

Create bowtie2 index:

```
bowtie2-build Mus_musculus.GRCm38.dna_sm.chromosome.12.fa Mus_musculus.GRCm38.dna_sm.chromosome.12
```



## Copy everything to shared directory

```bash
cd /work/projects/ulhpc-tutorials
mkdir -p bio/snakemake
cd bio/snakemake/
mkdir reference
mkdir chip-seq
cp /scratch/users/sdiehl/bioinfo_tutorial/reference/Mus_musculus.GRCm38.dna_sm.chromosome.7.fa* reference/
cp ~/chip-seq/*.fastq.gz chip-seq/
```

