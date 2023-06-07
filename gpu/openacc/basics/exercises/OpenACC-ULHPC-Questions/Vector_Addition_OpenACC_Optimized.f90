!! Author:Ezhilmathi Krishnasamy (ezhilmathi.krishnasamy@uni.lu)

module Vector_Addition_Mod                                                                                                                                                                                     
  implicit none                                                                                                                                                                                                
contains                                                                                                                                                                                                       
  subroutine Vector_Addition(a, b, c, n)                                                                                                                                                                       
    ! Input vectors                                                                                                                                                                                            
    real(8), intent(in), dimension(:) :: a                                                                                                                                                                     
    real(8), intent(in), dimension(:) :: b                                                                                                                                                                     
    real(8), intent(out), dimension(:) :: c                                                                                                                                                                    
    integer :: i, n                                                                                                                                                                                            
    !$acc data copyin(a(1:n), b(1:n)) copyout(c(1:n))                                                                                                                                                          
    !$acc parallel loop num_gangs(128) vector_length(128) 
    do i = 1, n
       c(i) = a(i) + b(i)
    end do
    !$acc end parallel
    !$acc end data                                                                                                                                                                                         
  end subroutine Vector_Addition                                                                                                                                                                               
end module Vector_Addition_Mod

program main
  use openacc
  use Vector_Addition_Mod
  implicit none
  
  ! Input vectors
  real(8), dimension(:), allocatable :: a
  real(8), dimension(:), allocatable :: b 
  ! Output vector
  real(8), dimension(:), allocatable :: c

  integer :: n, i                                                                   
  print *, "This program does the addition of two vectors "                         
  print *, "Please specify the vector size = "                                      
  read *, n  
  
  ! Allocate memory for vector
  allocate(a(n))
  allocate(b(n))
  allocate(c(n))
  
  ! Initialize content of input vectors, 
  ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
  do i = 1, n
     a(i) = sin(i*1D0) * sin(i*1D0)
     b(i) = cos(i*1D0) * cos(i*1D0) 
  enddo
    
  ! Call the vector add subroutine 
  call Vector_Addition(a, b, c, n)

  !!Verification
  do i = 1, n
     if (abs(c(i)-(a(i)+b(i))==0.00000)) then 
     else
        print *, "FAIL"
     endif
  enddo
  print *, "PASS"
  
  ! Delete the memory
  deallocate(a)
  deallocate(b)
  deallocate(c)
  
end program main
