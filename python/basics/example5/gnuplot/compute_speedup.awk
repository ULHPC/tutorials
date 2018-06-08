function speedup(t1, tp)
{
  return t1/tp;
}
function efficiency(t1, tp, p)
{
  return speedup(t1, tp)/p*100;
}

{
  if($1 == 1) {
    t1=$2;
  }
  print $1 " " speedup(t1,$2) " " efficiency(t1,$2,$1) > "data/time_per_core_speedup.dat";
}
