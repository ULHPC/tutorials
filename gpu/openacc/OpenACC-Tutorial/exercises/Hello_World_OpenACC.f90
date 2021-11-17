!! Author: Ezhilmathi Krishnasamy (ezhilmathi.krishnasamy@uni.lu)

subroutine Print_Hello_World()
  integer ::i
  !$acc kernels
  do i=1,5
     print *, "hello world"
  end do
  !$acc end kernels
end subroutine Print_Hello_World

program main
  use openacc
  implicit none
  call Print_Hello_World()
end program main


