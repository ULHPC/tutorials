`README.md`

Copyright (c) 2014 Joseph Emeras <joseph.emeras@uni.lu>

-------------------
# R Tutorial
Through this tutorial you will learn how to use R from your local machine or from one of the [UL HPC platform](http://hpc.uni.lu) clusters.
We will also use the `ggplot` library to generate nice graphics and export them as pdf files. 
Then, we will see how to organize and group data. Finally we will illustrate how R can benefit from multicore and cluster parallelization.

Warning: this tutorial does not focus on the learning of R language but aims at showing you nice startup tips.  
If you're also looking for a good tutorial on R's data structures you can take a look at: [Hadley Wickham's page](http://adv-r.had.co.nz/Data-structures.html).
 

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
For a complete list of availbale modules see: [Software page](https://hpc.uni.lu/users/software/).

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

    movies_url = "http://had.co.nz/data/movies/movies.tab.gz"								# this is the http url of data file
	## Download the file from given url to given destination file
	user_ = Sys.getenv("USER")
	dest_dir = paste0("/tmp/", user_)
	system(paste0("mkdir ", dest_dir))
	dest_file = paste0(dest_dir, "/movies.tab.gz")					
    download.file(movies_url, destfile=dest_file)											# we download the data file with `download.file()` function and save it in /tmp
	## Extract the .gz using system's gzip command
    system(paste0("gzip -d ", dest_file))													# `system()` function executes system calls
	dest_file = paste0(dest_dir, "/movies.tab")
	## Load the file in a dataframe with read.table() function
    movies = read.table(dest_file, sep="\t", header=TRUE, quote="", comment="")				# `read.table()` function reads a file and stores it in a data.frame object


Now let's take a (reproducible) sample of 1000 movies and plot their distribution regarding their rating.

    library(ggplot2)																		# load ggplot2 library to use packages functions
    set.seed(5689)																			# set the seed for random selection used in `sample()` function
    movies_sample = movies[sample(nrow(movies), 1000), ]									# movies is the data.frame name, from this data.frame, randomly select 1000 rows
    graph = ggplot(data=movies_sample) + geom_histogram(aes(x=rating), binwidth=0.5)		# construct the graph -- movies_sample will be used as data, we will plot an histogram where x=movies_sample$rating and with a bin size=0.5
    ggsave(graph, file="movies_hist.pdf", width=8, height=4)								# save the graph in a pdf file

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

So we have these two datasets, being of class dataframe. In R, a `data.frame` is one kind of data structure whose columns have names and that can contain several rows. Basically it looks like a matrix with columns identified by an index *and* a name, and with rows identified by an index.
Let's check how they are organized with the `names()` function that gives a dataset column names.

    > names(diamonds_fair)
	  [1] "carat" "price"
    
    > names(diamonds_good)
      [1] "carat" "price"

Thus for each dataset row we have the price and the carat value for a given diamond.
We want to add a column to datasets that will describe from which one it comes from, then we will merge these into one single dataset.

	diamonds_fair = cbind(diamonds_fair, cut_class="Fair")									# add a column named cut_class with all values being "Fair" to data.frame diamonds_fair
	diamonds_good = cbind(diamonds_good, cut_class="Good")									# same with "Good"
    diamonds_merge = rbind(diamonds_fair, diamonds_good)									# combine the 2 data.frame with `rbind()` as they both have the same structure

`cbind()` function is used to add a column to a dataframe, `rbind()` to combine rows of two dataframes (c is for column, r is for row).
Now we have all data merged in a dataframe and a column that describes the origin of data (the column `cut_class`), let's plot data.

Note: To visualize an extract of your data you can do:

    > diamonds_merge[1:10,]  		# returns rows 1 to 10
	> diamonds_merge[,3]     		# returns column no.3
	> diamonds_merge$cut_class		# returns column named cut_class

Then we construct and save the graph.

    graph = ggplot(data=diamonds_merge) + geom_point(aes(x=carat, y=price, colour=cut_class))	# this time we use ggplot's function `geom_point()` to plot data points. colour=cut_class aestetics option will plot the points according to cut_class values
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

	> ddply(diamonds, .(cut), summarize, avg_price=mean(price))				# in the data.frame named diamonds, aggregate by column named cut and apply the function mean() on the price of aggregated rows
	        cut avg_price
	1      Fair  4358.758
	2      Good  3928.864
	3 Very Good  3981.760
	4   Premium  4584.258
	5     Ideal  3457.542
will give us what we wanted.

Note: `ddply()` from the `plyr` package is similar to `aggregate()` from base package, you can use indifferently one or the other, `plyr` functions simply provide a more consistent naming convention.



### Perfomance Considerations
In the previous section for the aggregation, instead of using `ddply`, we could also have used `lapply` (but in a slightlier more complicated way):

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
	png("benchmark_boxplot.png")										# other method to save graphics that are not generated with ggplot. We give a name to the output graphic
	## plot the graph
	plot(m)																# then we plot it
	## flush the output device to save the graph
	dev.off()															# finally we close the output device, this will save the graphic in the output file


### Using `data.table` Package

According to [data.table documentation](http://cran.r-project.org/web/packages/data.table/index.html) `data.table` inherits from `data.frame` to offer fast subset, fast grouping, fast update, fast ordered
joins and list columns in a short and flexible syntax, for faster development. It uses binary search instead of vector scan to perform its operations and thus is scalable.
We can convert easily a `data.frame` to a `data.table`:

	> MOVIES = data.table(movies)

As `data.table` uses binary search, we have to define manually the keys that will be used for this search, this is done with `setkey()` function.

Let's now create a new `data.frame`. We will make it large enough to demonstrate the difference between a vector scan and a binary search.

    grpsize = ceiling(1e7/26^2) # 10 million rows, 676 groups
    system.time( DF <- data.frame(
		x=rep(LETTERS,each=26*grpsize),
		y=rep(letters,each=grpsize),
		v=runif(grpsize*26^2),
		stringsAsFactors=FALSE)
	)	

This generated a data.frame named DF with 3 columns. Column x is a repetition of uppercase letters from A to Z, column y is minorcase letters. Column v is a random uniform value.
To illustrate the difference, we take as example the selection in this large dataset of rows where x=="R" and y=="h".

	> system.time(ans1 <- DF[DF$x=="R" & DF$y=="h",]) 		# vector scan. we select rows where x="R" and y="h". For this we have to scan the full data.frame twice.
	
	> DT = data.table(DF)									# convert the data.frame to a data.table
	> setkey(DT,x,y)										# set column x and y as data.table keys.
	> system.time(ans2 <- DT[J("R","h")]) 					# binary search. We select rows that match the join between DT and the data.table row: data.table("R","h"). This will return the same result as before but much faster.

In the first case, we scan the full table twice (once for selecting x's that are equal to "R", then y's that are equal to "h"), then do the selection.
In the second case, we are joining DT to the 1 row, 2 column table returned by data.table("R","h"). We use the alias for joining data.tables called J(), short for join. As we defined x and y as keys, this works like a database join.
You can see that vector scan is very long compared to binary search.

#### Grouping
`data.table` also provides faster operations for reading files and grouping data.

Now you can compare the same aggregation operation with `data.frame` and `data.table`. In both examples we aggregate on x and apply the function `sum()` to corresponding v.


`data.frame` style:

	system.time(tapply(DT$v,DT$x,sum))

`data.table` style, using `by`:

	system.time(DT[,sum(v),by=x])


**Question: use `ddply()` instead of `tapply()` in the first example.**

<!--
ddply(DT, .(x), summarize, sum(v))
-->

**Question: return the min and max instead of the sum.**

Hint: you can create a function named min_max to help you doing this. Example:

	dummy_function = function(x){ x }					# dummy_function(x) will return x.
	dummy_function2 = function(x, y){ c(x, y) }			# dummy_function2(x, y) will return a vector (x,y).

<!--
min_max = function(data){
  c(min(data), max(data))
}
DT[,min_max(v),by=x]
	
or

DT[,c(min(v), max(v)),by=x]	
-->

## Parallel R
The first part of the tutorial is now over, you can connect to `gaia` cluster and submit an other job requesting several machines.

	jdoe@localhost:~$ ssh gaia-cluster
	
    jdoe@access:~$ oarsub -I -l nodes=2,walltime=1

<!--
When the job is running and you are connected load R module (version compiled with GCC).

    jdoe@access:~$ module load R/3.0.2-goolf-1.4.10
-->

When the job is running and you are connected load R module (version compiled with Intel Compiler), then run R.

    jdoe@access:~$ module load R/3.0.2-ictce-5.3.0
	jdoe@access:~$ R


We will use a large dataset (400K+ rows) to illustrate the effect of parallelization in R (as dataset is large, the following line may take time to complete depending on your network speed).

    > air = read.csv(url("http://packages.revolutionanalytics.com/datasets/AirOnTimeCSV2012/airOT201201.csv"))

**NOTE**: If downloading the air dataset (above line) takes too much time you can load it from a file on the cluster:

	> load("~jemeras/data/air.rda")
 
If we want to have the number of flights for each destination `DEST` we can do the following:

    dests = as.character(unique(air$DEST))
	count_flights = function(x){length(which(air$DEST == x))}
	as.data.frame(cbind(dest=dests, nb=lapply(dests, count_flights)))

As the dataframe is large it takes some time to compute

	> microbenchmark(LAPPLY=lapply(dests, count_flights), times=10)
	Unit: seconds
	   expr      min       lq   median       uq      max neval
	 LAPPLY 1.607961 1.609036 1.609638 1.610269 2.023961    10

### Single Machine Parallelization
To parallelize the lapply function we can use `mclapply()` from `multicore` package and give it the number of cores to use.
`mclapply()` uses the underlying operating system fork() functionality to achieve parallelization.
Using several cores makes the process shorter.

    > library(multicore)
    > as.data.frame(cbind(dest=dests, nb=mclapply(dests, count_flights, mc.cores=12)))
	
	
	> microbenchmark(MCLAPPLY=mclapply(dests, count_flights, mc.cores=12), times=10)  # or use `detectCores()` from `parallel` package instead of giving cores value. 
	Unit: milliseconds
	     expr      min       lq   median       uq     max neval
	 MCLAPPLY 233.8035 235.1089 235.9138 236.6393 263.934    10
    

You can now save the `air` R object to reuse it in an other R session.

    > save(air, file="./air.rda")

Then quit your current R session but **do not** end your current oar job.

### Cluster Parallelization
The `parLapply()` function will create a cluster of processes, which could even reside on different machines on the network, and they communicate via TCP/IP or MPI in order to pass the tasks and results between each other.
Thus you have to load necessary packages and export necessary data and functions to the global environment of the cluster workers.


First, load R 3.0.2 compiled with GCC as Intel one does not work for this. Add the module loading at bash login too for enabling it on the nodes. To do so, within a shell type:

    echo 'module load R/3.0.2-goolf-1.4.10' >> ~/.bash_login		# /!\ TO REMOVE AT THE END OF PS /!\
	module purge
	module load R/3.0.2-goolf-1.4.10

**Warning: do not forget to clean your ~/.bash_login file after the PS (remove the 'module load R/3.0.2-goolf-1.4.10' line).**

#### Socket Communications

First, let's load data and initialize variables.

    library(parallel)
	
	# air = read.csv(url("http://packages.revolutionanalytics.com/datasets/AirOnTimeCSV2012/airOT201201.csv"))  # load again the air data.table from http
    load("./air.rda")	# read saved R object from file "air.rda"
	dests = as.character(unique(air$DEST))
	count_flights = function(x){length(which(air$DEST == x))}
	
	## set cluster characteristics -- get OAR nodes to use, type of communication
	nodes = scan(Sys.getenv("OAR_NODE_FILE"), what=character(0))
	oar_job_id = as.numeric(Sys.getenv("OAR_JOB_ID"))
	connector = paste0("OAR_JOB_ID=", oar_job_id)
	connector = paste0(connector, " oarsh")
	comm_type = "PSOCK"
	
Then, setup the cluster.
	
	## set up the cluster
	cl = makeCluster(nodes, type = comm_type, rshcmd = connector)	
	## If a particular library <LIB> is needed, load it on the nodes with
	# clusterEvalQ(cl, { library(<LIB>) })
	## or give the full environment with
	# clusterEvalQ(cl, sessionInfo())
	## export air dataset on all the nodes
	clusterExport(cl, varlist=c("air"))
	
Do the parallel computation.
	
	## compute in parallel through sockets
	as.data.frame(cbind(dest=dests, nb=parLapply(cl, dests, count_flights)))
	
Finalize and cleanup things.
	
	stopCluster(cl)
 


**Exercise: Plot a speedup graph with different number of cores and/or machines used.**


#### Not (yet) Covered by this Tutorial: MPI Communications

It is also possible to use MPI communications instead of sockets.
We will not see this in the tutorial because some required modules will be available in the platform in a close future, however here is the basic procedure.

R will need the package `Rmpi` and same as before, we use `makeCluster` but we use `comm_type = "MPI"` instead of `PSOCK` and we call `mpi.exit()` after calling `stopCluster()`.
Then, you need to call the R script within MPI. i.e. 
   
    mpirun -n <nb_processes> R --slave -f <R_script_file.R>


### Usefull links

* [CRAN Archive](http://cran.r-project.org/)

* [CRAN HPC Packages](http://cran.r-project.org/web/views/HighPerformanceComputing.html)

* [ggplot2 Documentation](http://docs.ggplot2.org/current/)

* [Advanced R programming by Hadley Wickham](http://adv-r.had.co.nz/)


