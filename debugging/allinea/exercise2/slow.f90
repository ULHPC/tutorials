program slow
use mpi
implicit none
integer :: pe, nprocs, ierr

call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, pe, ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)

if (pe == 0) print *,"Starting multiplication"
call multiplication
if (pe == 0) print *,"Starting square root"
call sqrroot

call MPI_FINALIZE(ierr)

contains

subroutine sqrroot

  integer :: i,j,iterations
  real    :: a(6000),b(6000)

  do iterations=1,4
    a=1.1 + iterations
    do j=0,pe
      do i=1,size(a)
         a=sqrt(a)+1.1*j
      end do
    end do
    call MPI_ALLREDUCE(a,b,size(a),MPI_REAL,MPI_SUM,MPI_COMM_WORLD,ierr)
  end do
  if (pe == 0) print *,"sqrroot answer",b(1)
  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

end subroutine sqrroot

subroutine multiplication

  implicit none
  real :: a(2000,2000)
  integer :: i,j,l
  real :: x,y
  do l=1,200
    do j=1,2000
      do i=1,2000
        x=i
        y=j
        a(i,j)=x*j
      end do
    end do
  end do

  if (pe == 0) print *,"multiplication answer",sum(a)
  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

end subroutine multiplication

end program slow

