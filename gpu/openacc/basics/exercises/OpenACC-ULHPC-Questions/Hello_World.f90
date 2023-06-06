!! Authour:Ezhilmathi Krishnasamy (ezhilmathi.krishnasamy@uni.lu)

subroutine Print_Hello_World()
  integer ::i
  do i=1,5
     print *, "hello world"
  end do
end subroutine Print_Hello_World

program main
  implicit none
  call Print_Hello_World()
end program main


