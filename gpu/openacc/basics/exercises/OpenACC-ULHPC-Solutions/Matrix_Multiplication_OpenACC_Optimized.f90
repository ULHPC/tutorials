!! Authour:Ezhilmathi Krishnasamy (ezhilmathi.krishnasamy@uni.lu)

module Matrix_Multiplication_Mod  
  implicit none 
contains
  subroutine Matrix_Multiplication(a, b, c, width)
    use openacc
    ! Input vectors
    real(8), intent(in), dimension(:) :: a
    real(8), intent(in), dimension(:) :: b
    real(8), intent(out), dimension(:) :: c
    real(8) :: sum = 0
    integer :: i, row, col, width
    !$acc parallel num_gangs(64 )vector_length(64) copyin(a(1:width*width), b(1:width*width)) copyout(c(1:width*width)) create(sum)
    !$acc loop collapse(2) reduction(+:sum) 
    do row = 0, width-1
       do col = 0, width-1
          sum=0
          do i = 0, width-1
             sum = sum + (a((row*width)+i+1) * b((i*width)+col+1))
          enddo
          c(row*width+col+1) = sum
       enddo
    enddo
    !$acc end loop
    !$acc end parallel
    
  end subroutine Matrix_Multiplication
end module Matrix_Multiplication_Mod

program main
  use Matrix_Multiplication_Mod
  use openacc
  implicit none
  
  ! Input vectors
  real(8), dimension(:), allocatable :: a
  real(8), dimension(:), allocatable :: b
  
  ! Output vector
  real(8), dimension(:), allocatable :: c
  ! real(8) :: sum = 0
  
  integer :: n, i 
  print *, "This program does the addition of two vectors "
  print *, "Please specify the vector size = "
  read *, n
  
  ! Allocate memory for vector
  allocate(a(n*n))
  allocate(b(n*n))
  allocate(c(n*n))

  ! Initialize content of input vectors, 
  ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
  do i = 1, n*n
     a(i) = 1.0 !!!sin(i*1D0) * sin(i*1D0)
     b(i) = 2.0 !!!cos(i*1D0) * cos(i*1D0) 
  enddo

  ! Call the vector add subroutine 
  call Matrix_Multiplication(a, b, c, n)

  !!Verification
  do i=1,n*n
     print *, c(i)
  enddo

  ! Delete the memory
  deallocate(a)
  deallocate(b)
  deallocate(c)

end program
