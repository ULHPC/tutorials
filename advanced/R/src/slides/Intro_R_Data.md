##R?
R (pronounced aRrgh -- pirate style) is a programming language and environment for statistical computing and graphics

- oriented towards data handling analysis and storage facility

- R Base

- Packages tools and functions (user contributed)

- R Base and most R packages are available from the [Comprehensive R Archive Network (CRAN)](cran.r-project.org)

- Use R console or IDE: **Rstudio**, Deducer, vim/emacs...

- Comment is **#**, help is **?** before a function name


## Using R
###**Installing/using packages**
Install and load the `ggplot2` package (even if already installed)

	install.packages("ggplot2")
	library(ggplot2)


Or in one step, install if not available then load:

	require(ggplot2) || {install.packages("ggplot2");
	 					 require(ggplot2)}

## Using R
###**Usefull Functions**

- List all objects in memory: `ls()`

- Save an object: `save(obj, file)`

- Load an object: `load(file)`

- Set working directory: `setwd(dir)`

##Data Structures

- scalar:

	s = 3.14

- vector:

	v = c(1, 2, "ron")

- list: 

	l = list(1:10, 'a', pi)

- matrix:

	m = matrix(seq(1:6), 2)

- **dataframe**: 

	df = data.frame("col1" = seq(1:4), "col2" = c(5, 6, "cerveza", 6*7))

- ...

##Entering Data
### Reading CSV or text files

	# comma separated values
	dat.csv <- read.csv(<file or url>)
	# tab separated values
	dat.tab <- read.table(<file or url>, 
		header=TRUE, sep = "\t")

##Entering Data
### Reading data from other software: Excel, SPSS...

Excel Spreadsheets -- need `xlsx` package

	read.xlsx()
	

SPSS and Stata both need the `foreign` package

	dat.spss <- read.spss(<file or url>, 
				 		  to.data.frame=TRUE)
				 
	dat.dta <- read.dta(<file or url>)


## Data Frames
Most easy structure to use, have a matrix structure. 

- **Observations** are arranged as **rows** and **variables**, either numerical or categorical, are arranged as **columns**.

- Individual rows, columns, and cells in a data frame can be accessed through many methods of indexing.

- We most commonly use **object[row,column]** notation.


## Accessing Items in a `data.frame`

Aside with R are provided example datasets, i.e. `mtcars` that can be used

	data(mtcars)
	head(mtcars)
	colnames(mtcars)

	# single cell value
	mtcars[2,3]
	# omitting row value implies all rows
	mtcars[,3]
	# omitting column values implies all columns
	mtcars[2,]



## Accessing Items in a `data.frame`
We can also access variables directly by using their names, either with **object[,"variable"]** notation or **object$variable** notation.

	# get first 10 rows of variable `mpg` using two methods:
	mtcars[1:10, "mpg"]
	mtcars$mpg[1:10]


##Exploring Data
###Description Of Dataset
- Using **dim**, we get the number of observations(rows) and variables(columns) in the dataset.

- Using **str**, we get the structure of the dataset, including the class(type) of all variables.

	dim(mtcars)
	str(mtcars)

- **summary** when used on a dataset, returns distributional summaries of variables in the dataset.

	summary(mtcars)

- **quantile** function enables to get statistical metrics on the selected data

	quantile(mtcars$mpg)
	
##Exploring Data
###Conditional Exploration
	
- **subset** enables to explore data conditionally 

	subset(mtcars, cyl <= 5)
	
- **by** enables to call a particular function to sub-groups of data

	by(mtcars, mtcars$cyl, summary)
	
