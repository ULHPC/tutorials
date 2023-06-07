!! Authour: Ezhilmathi Krishnasamy (ezhilmathi.krishnasamy@uni.lu)

subroutine Print_Hello_World()
  integer ::i
  !$acc ......????? 
  do i=1,150
     print *, "hello world"
  end do
  !$acc ......???? 
end subroutine Print_Hello_World

program main
  !!use OPENACC LIBRARY ???? 
  implicit none
  call Print_Hello_World()
end program main


