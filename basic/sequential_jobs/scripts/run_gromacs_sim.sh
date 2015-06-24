#!/bin/bash -l

VALUE=$1
SRC=ftp://ftp.gromacs.org/pub/benchmarks/rnase_bench_systems.tar.gz

cd /tmp

# extract the example input files

if [[ ! -d rnase_cubic ]] ; then
  wget $SRC
  tar xzvf rnase_bench_systems.tar.gz rnase_cubic
fi

# Create a new workdir

cp -R rnase_cubic /tmp/rnase_cubic_fs_$VALUE
cd /tmp/rnase_cubic_fs_$VALUE

# Load Gromacs

module load bio/GROMACS/4.6.5-goolf-1.4.10-mt

# Generate a parameter file based on the value of the first parameter of this script

sed -i "s/^fourier_spacing.*$/fourier_spacing=${VALUE}/" pme_verlet.mdp
grompp -f pme_verlet.mdp -c conf.gro -p topol.top -o bench_rnase_cubic.tpr
mdrun -nt 1 -s bench_rnase_cubic.tpr -nsteps 1000

