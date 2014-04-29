`README.md`

Copyright (c) 2014 Joseph Emeras <joseph.emeras@uni.lu>

-------------------
# R Tutorial
Through this tutorial you will learn how to use R from your local machine or from one of the [UL HPC platform](http://hpc.uni.lu) clusters.
We will also use the `ggplot` library to generate nice graphics and export them as pdf files. 
Then, we will see how to organize and group data. Finally we will illustrate how R can benefit from multicore and cluster parallelization.

## Pre-requisites

### Optional: On your local machine
First of all, let's install R. You will find releases for various distributions available at [CRAN Archive](http://cran.r-project.org/).
Once installed, to use R interactive session interface, simply open a terminal and type:

    jdoe@localhost:~$ R

You will also find handy to use the [R-Studio](https://www.rstudio.com/) graphical IDE. R-Studio embeds a R shell where you can call R functions as in the interactive session interface.
Thus you can use whether R interactive shell or R-Studio embedded shell.

### On the cluster
R is already available in `Chaos` and `Gaia` clusters as a module. 
The first step is the reservation of a resource. Connect to your favorite cluster frontend (here: `chaos`)

    jdoe@localhost:~$ ssh chaos-cluster

Once connected to the user frontend, book 1 core for half an hour (as we will use R in single-threaded mode, we will need only one core).

    jdoe@access:~$ oarsub -I -l core=1,walltime="00:30:00"

When the job is running and you are connected load R module (version compiled with Intel Compiler).

    jdoe@access:~$ module load R/3.0.2-ictce-5.3.0

<!--
    jdoe@access:~$ module load R/3.0.2-goolf-1.4.10
-->
Now you should be able to invoke R and see something like this:

    jdoe@cluster-node-1:~$ R
    R version 3.0.2 (2013-09-25) -- "Frisbee Sailing"
    Copyright (C) 2013 The R Foundation for Statistical Computing
    Platform: x86_64-unknown-linux-gnu (64-bit)
    
    R is free software and comes with ABSOLUTELY NO WARRANTY.
    You are welcome to redistribute it under certain conditions.
    Type 'license()' or 'licence()' for distribution details.
    
    R is a collaborative project with many contributors.
    Type 'contributors()' for more information and
    'citation()' on how to cite R or R packages in publications.
    
    Type 'demo()' for some demos, 'help()' for on-line help, or
    'help.start()' for an HTML browser interface to help.
    Type 'q()' to quit R.
    
    >


### Installing R Packages
`sessionInfo()` function gives information about R version, loaded libraries etc.

    > sessionInfo()
    R version 3.0.2 (2013-09-25)
    Platform: x86_64-unknown-linux-gnu (64-bit)
    
    locale:
    [1] C
    
    attached base packages:
    [1] stats     graphics  grDevices utils     datasets  methods   base
    >


To install libraries you can use the `install.packages()` function. e.g
<!---
    > install.packages("Rcpp")
	> install.packages("plyr")
-->

	> install.packages("ggplot2")

This will install the `ggplot2` library.

Note: on the first run, R might ask you various questions during the installation. e.g. selecting a CRAN mirror to use for downloading packages. Select a mirror close to your location. For other questions, using default values is ok.

Now, to load this library call:

    > library(ggplot2)

A call to `sessionInfo()` function will return `ggplot2` version as it is now attached to the current session.



## Warm-up Session -- Simple Plotting

### From Single Dataset 
Movies dataset, derived from data provided by [IMDB](http://imdb.com) is a sample dataset available in `ggplot2` for testing purpose. Its data description can be found [here](http://had.co.nz/data/movies/description.pdf).
Thus, when loading `ggplot2` library, this dataset is available under the name: `movies`.

(OPTIONAL) An other way to get the dataset would be to download, extract and read it with:

    movies_url = "http://had.co.nz/data/movies/movies.tab.gz"
	## Download the file from given url to given destination file
    download.file(movies_url, destfile="/tmp/movies.tab.gz")
	## Extract the .gz using system's gzip command
    system("gzip -d /tmp/movies.tab.gz")
	## Load the file in a dataframe with read.table() function
    movies = read.table("/tmp/movies.tab", sep="\t", header=TRUE, quote="", comment="")


Now let's take a (reproducible) sample of 1000 movies and plot their distribution regarding their rating.

    library(ggplot2)
    set.seed(5689)
    movies_sample = movies[sample(nrow(movies), 1000), ]
    graph = ggplot(data=movies_sample) + geom_histogram(aes(x=rating), binwidth=0.5)
    ggsave(graph, file="movies_hist.pdf", width=8, height=4)

Now you shoud be able to visualize the generated plot on the frontend (may not work on windows stations using PuTTy):

    jdoe@access:~$ epdfview movies_hist.pdf

`ggplot2` proposes many functions to plot data according to your needs. Do not hesitate to wander in the [ggplot2 documentation](http://docs.ggplot2.org/current/) and to read at provided examples to better understand how to use it.
The `ggsave()` function is convenient to export ggplot graphics as .pdf or .png files

### From Several Datasets
Now, let's say we have two different datasets: `diamonds_fair` and `diamonds_good` that are both extracts from the `diamonds` dataset (also provided in ggplot2).
In this example we will consider that these two datasets come from different sources, so do not try to understand the next lines, they are just here to setup the example (simply copy-paste these in your R prompt).

    set.seed(2109)	
    diamonds_fair = data.frame(carat=diamonds$carat[which(diamonds$cut == 'Fair')], price=diamonds$price[which(diamonds$cut == 'Fair')])
	diamonds_fair = diamonds_fair[sample(nrow(diamonds_fair), 20), ]
	diamonds_good = data.frame(carat=diamonds$carat[which(diamonds$cut == 'Good')], price=diamonds$price[which(diamonds$cut == 'Good')])
	diamonds_good = diamonds_good[sample(nrow(diamonds_good), 20), ]

To know the class of an R object you can use the `class()` function

    > class(diamonds_fair)
      [1] "data.frame"

So we have these two datasets (being of class dataframe), let's check how they are organized with the `names()` function that gives a dataset column names.

    > names(diamonds_fair)
	  [1] "carat" "price"
    
    > names(diamonds_good)
      [1] "carat" "price"

Thus for each dataset row we have the price and the carat value for a given diamond.
We want to add a column to datasets that will describe from which one it comes from, then we will merge these into one single dataset.

	diamonds_fair = cbind(diamonds_fair, cut_class="Fair")
	diamonds_good = cbind(diamonds_good, cut_class="Good")
    diamonds_merge = rbind(diamonds_fair, diamonds_good)

`cbind()` function is used to add a column to a dataframe, `rbind()` to combine rows of two dataframes (c is for column, r is for row).
Now we have all data merged in a dataframe and a column that describes the origin of data (the column `cut_class`), let's plot data.

    graph = ggplot(data=diamonds_merge) + geom_point(aes(x=carat, y=price, colour=cut_class))
	ggsave(graph, file="diamonds_plot.pdf", width=8, height=4)

Remember, to get help about a particular function you can type `?function_name`. e.g.

    > ?cbind

To get package and meta information on a function you can type `??function_name`. e.g.

    > ??ggsave



## Organizing your Data

Let's say we are working with the full `diamonds` dataset and we want to have the average price for a given diamond cut. 

    > names(diamonds)
     [1] "carat"   "cut"     "color"   "clarity" "depth"   "table"   "price"
     [8] "x"       "y"       "z"

We could do a for loop to aggregate the data per cuts and manually compute the average price, but in R loops are generally a bad idea. For large datasets it is very long to compute.
Thus instead of looping around the dataset, we will use a function from the `plyr` package: `ddply()`. 
You will first need to install and load `plyr`.

    install.packages("plyr")
	library(plyr)
Now we are ready to call `ddply()`. The first parameter will be the dataset, the second will be the column of the dataset we want to aggregate on, third parameter will be the call to the `summarize()` function that will enable to aggregate data on the `cut` column. The forth parameter will be the operation we will do for each of the aggregated classes. Thus: 

	> ddply(diamonds, .(cut), summarize, avg_price=mean(price))
	        cut avg_price
	1      Fair  4358.758
	2      Good  3928.864
	3 Very Good  3981.760
	4   Premium  4584.258
	5     Ideal  3457.542
will give us what we wanted.

Note: `ddply()` from the `plyr` package is similar to `aggregate()` from base package, you can use indifferently one or the other, `plyr` functions simply provide a more consistent naming convention.



### Perfomance Considerations
In the previous section for the aggregation, instead of using `ddply`, we could also have used `lapply` (but I recognize in a slightlier more complicated way):

    > as.data.frame(cbind(cut=as.character(unique(diamonds$cut)), avg_price=lapply(unique(diamonds$cut), function(x) mean(diamonds$price[which(diamonds$cut == x)]))))

So, we want to know which one of the two versions is the most efficient, for that purpose, the library `microbenchmark` is handy.

    install.packages("microbenchmark")
	library(microbenchmark)

We can use the `microbenchmark()` function on several expressions, with a given repetition number to compare them:

	> m = microbenchmark(DDPLY=ddply(diamonds, .(cut), summarize, avg_price=mean(price)), LAPPLY=as.data.frame(cbind(cut=as.character(unique(diamonds$cut)), avg_price=lapply(unique(diamonds$cut), function(x) mean(diamonds$price[which(diamonds$cut == x)])))), times=1000)
	> m
	Unit: milliseconds
	   expr      min       lq   median       uq       max neval
	  DDPLY 24.73218 29.10263 65.50023 69.80662 140.54594  1000
	 LAPPLY 22.85223 24.44387 25.55315 27.45517  96.94869  1000

Plotting the benchmark result gives us a boxplot graph:

    ## save the output graph as png file
	png("benchmark_boxplot.png")
	## plot the graph
	plot(m)
	## flush the output device to save the graph
	dev.off()


### Using `data.table` Package

According to [data.table documentation](http://cran.r-project.org/web/packages/data.table/index.html) `data.table` inherits from `data.frame` to offer fast subset, fast grouping, fast update, fast ordered
joins and list columns in a short and flexible syntax, for faster development. It uses binary search instead of vector scan to perform its operations and thus is scalable.
We can convert easily a `data.frame` to a `data.table`.

	> MOVIES = data.table(movies)

Because `data.table` uses binary search, we have to define manually the keys that will be used for this search.

Let's now create a new `data.frame`. We will make it large enough to demonstrate the difference between a vector scan and a binary search.

    grpsize = ceiling(1e7/26^2) # 10 million rows, 676 groups
    system.time( DF <- data.frame(
		x=rep(LETTERS,each=26*grpsize),
		y=rep(letters,each=grpsize),
		v=runif(grpsize*26^2),
		stringsAsFactors=FALSE)
	)	


Vector Scan is long compared to binary search

	> system.time(ans1 <- DF[DF$x=="R" & DF$y=="h",]) # vector scan
	
	> DT = data.table(DF)
	> setkey(DT,x,y)
	> system.time(ans2 <- DT[J("R","h")]) # binary search	

In this case, we are joining DT to the 1 row, 2 column table returned by data.table("R","h"). We use the alias for joining data.tables called J(), short for join.


#### Grouping

DataFrame style:

	system.time(tapply(DT$v,DT$x,sum))

`data.table` style, using `by`:

	system.time(DT[,sum(v),by=x])




## Parallel R
The first part of the tutorial is now over, you can connect to `gaia` cluster and submit an other job requesting several machines.

	jdoe@localhost:~$ ssh gaia-cluster
	
    jdoe@access:~$ oarsub -I -l nodes=2,walltime=1

When the job is running and you are connected load R module (version compiled with GCC).

    jdoe@access:~$ module load R/3.0.2-goolf-1.4.10


We will use a large dataset (400K+ rows) to illustrate the effect of parallelization in R.

    > air = read.csv(url("http://packages.revolutionanalytics.com/datasets/AirOnTimeCSV2012/airOT201201.csv"))

If we want to have the number of flights for each destination `DEST` we can do the following:

    dests = as.character(unique(air$DEST))
	count_flights = function(x){length(which(air$DEST == x))}
	as.data.frame(cbind(dest=dests, nb=lapply(dests, count_flights)))

As the dataframe is large it takes some time to compute

	> microbenchmark(LAPPLY=lapply(dests, count_flights), times=10)
	Unit: seconds
	   expr      min       lq   median       uq      max neval
	 LAPPLY 10.84337 10.97527 11.06003 11.08972 11.37946    10

### Single Machine Parallelization
To parallelize the lapply function we can use `mclapply()` from `multicore` package and give it the number of cores to use.
`mclapply()` uses the underlying operating system fork() functionality to achieve parallelization.
Using several cores makes the process shorter.

    > library(multicore)
    > as.data.frame(cbind(dest=dests, nb=mclapply(dests, count_flights, mc.cores=4)))
	
	
	> microbenchmark(MCLAPPLY=mclapply(dests, count_flights, mc.cores=4), times=10)  # or use `detectCores()` from `parallel` package instead of giving cores value. 
	Unit: seconds
	     expr      min       lq   median       uq      max neval
	 MCLAPPLY 4.353838 4.373936 4.525893 4.982022 5.076462    10
    

### Cluster Parallelization
The `parLapply()` function will create a cluster of processes, which could even reside on different machines on the network, and they communicate via TCP/IP or MPI in order to pass the tasks and results between each other.
Thus you have to load necessary packages and export necessary data and functions to the global environment of the cluster workers.


First, load R 3.0.2 module at bash login. To do so, within a shell type:

    echo 'module load R/3.0.2-goolf-1.4.10' >> ~/.bash_login

#### Socket Communications
    library(parallel)
	
	air = read.csv(url("http://packages.revolutionanalytics.com/datasets/AirOnTimeCSV2012/airOT201201.csv"))
    dests = as.character(unique(air$DEST))
	count_flights = function(x){length(which(air$DEST == x))}
	
	## set cluster characteristics -- get OAR nodes to use, type of communication
	nodes = scan(Sys.getenv("OAR_NODE_FILE"), what=character(0))
	oar_job_id = as.numeric(Sys.getenv("OAR_JOB_ID"))
	connector = paste0("OAR_JOB_ID=", oar_job_id)
	connector = paste0(connector, " oarsh")
	comm_type = "PSOCK"
	## set up the cluster
	cl = makeCluster(nodes, type = comm_type, rshcmd = connector)	
	## If a particular library <LIB> is needed, load it on the nodes with
	# clusterEvalQ(cl, { library(<LIB>) })
	## or give the full environment with
	# clusterEvalQ(cl, sessionInfo())
	## export air dataset on all the nodes
	clusterExport(cl, varlist=c("air"))
	## compute in parallel through sockets
	as.data.frame(cbind(dest=dests, nb=parLapply(cl, dests, count_flights)))
	stopCluster(cl)


Note1: You can measure the time taken by a function with `system.time()`

Note2: On a single node, `parLapply()` is slightly more efficient than `mclapply()` but datasets need to be exported before (may take some time).

	> microbenchmark(PARLAPPLY=parLapply(cl, dests, count_flights), times=10)
	Unit: seconds
	      expr      min       lq   median      uq      max neval
	 PARLAPPLY 4.287688 4.301338 4.438636 4.59001 4.741495    10


#### MPI Communications
Same as before but we use `comm_type = "MPI"` and call `mpi.exit()` after calling `stopCluster()`.
To run MPI version you need to call the R script within MPI. i.e. 
   
    mpirun -n <nb_processes> R --slave -f <R_script_file.R>


TODO:

    module load OpenMPI/1.7.3-GCC-4.8.2
    ## In R:
	# install.packages("Rmpi")





### Usefull links


http://cran.r-project.org/web/views/HighPerformanceComputing.html






