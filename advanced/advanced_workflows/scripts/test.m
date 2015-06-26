N=1000;
fib=zeros(1,N);
fib(1)=1;
fib(2)=1;
k=3;
while k <= N
  fib(k)=fib(k-2)+fib(k-1);
  fprintf('%d\n',fib(k));
  pause(1);
  k=k+1;
end
