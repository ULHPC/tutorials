MPICC=mpicc
MPIF90=mpif90
MPIRUN=mpirun

all:
# gnu
	${MPIF90} -g -O3 -fno-inline slow.f90 -o slow_f -lm -lrt
# intel
#       ${MPIF90} -g -fno-inline-functions -O3 slow.f90 -o slow_f -lm -lrt
# pgi
#       ${MPIF90} -g -Mprof=func -O3 slow.f90 -o slow_f -lm -lrt

check:
	${MPIRUN} -np 4 ./slow_f 3

clean:
	-rm -f ./slow_f
