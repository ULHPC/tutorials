len <- 130
fibvals <- numeric(len)
fibvals[1] <- 1
fibvals[2] <- 1
for (i in 3:len) {
     fibvals[i] <- fibvals[i-1]+fibvals[i-2]
     print( fibvals[i], digits=22)
     Sys.sleep(2)
}

