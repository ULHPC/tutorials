`README.md`

Copyright (c) 2014 Valentin Plugaru <Valentin.Plugaru@gmail.com>

-------------------


# UL HPC Tutorial: MATLAB execution on the UL HPC platform

The objective of this tutorial is to show the execution of [Matlab](http://www.matlab.com) - 
a high-level language and interactive environment for numerical computation, 
visualization and programming, on top of the [UL HPC](http://hpc.uni.lu) platform.

## Prerequisites

As part of this tutorial two Matlab example scripts have been developed and you will need to download them,
along with their dependencies, before following the instructions in the next sections:

        (gaia-frontend)$> mkdir ~/matlab-tutorial
        (gaia-frontend)$> cd ~/matlab-tutorial
        (gaia-frontend)$> wget https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB/example1.m
        (gaia-frontend)$> wget https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB/example2.m
        (gaia-frontend)$> wget https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB/yahoo_finance_data.m

## Matlab execution in interactive mode

### Launching the full graphical environment

Running the full MATLAB environment (e.g. on the Gaia cluster) requires an [interactive OAR session](https://hpc.uni.lu/users/docs/oar.html#concepts). When connecting to the clusters you will
need to enable X11 forwarding in order for the graphical environment to be shown on your 
local machine:

- on Linux and OS X simply follow the commands below

- on Windows you will need to run [XMing](http://sourceforge.net/projects/xming/) first
  then to configure Putty (Connection -> SSH -> X11 -> Enable X11 forwarding) before
  logging in to the clusters.

        # Connect to Gaia with X11 forwarding enabled (Linux/OS X):
        (yourmachine)$> ssh access-gaia.uni.lu -X
     
        # Request an interactive job (the default parameters get you 1 core for 2 hours):
        (gaia-frontend)$> oarsub -I

        # Check the Matlab versions installed on the clusters:
        (node)$> module available 2>&1 | grep -i matlab
        
        # Load a specific MATLAB version:
        (node)$> module load MATLAB/2013a

        # Check that it has been loaded, along with Java:
        (node)$> module list
        
        # Launch MATLAB
        (node)$> matlab
        
After a delay, the full Matlab interface will be displayed on your machine and you will be able to run commands, load and edit
scripts and generate plots. An alternative to the graphical interface is the command-line (text-mode) interface, which is
enabled through specific parameters, described in the following section.

### Launching the command-line environment

Running the text-mode MATLAB interface in an interactive session, is much faster than
using the full graphical environment through the network and is useful for commands/scripts testing and 
quick executions:

        # First, connect to an UL cluster (e.g. Gaia):
        
        (yourmachine)$> ssh access-gaia.uni.lu
        (gaia-frontend)$> oarsub -I
        (node)$> module load MATLAB/2013a
     
        # Launch MATLAB with the graphical display mode disabled (critical parameters):
        (node)$> matlab -nodisplay -nosplash
        Opening log file:  /home/users/vplugaru/java.log.3258
                                                                        < M A T L A B (R) >
                                                              Copyright 1984-2013 The MathWorks, Inc.
                                                                R2013a (8.1.0.604) 64-bit (glnxa64)
                                                                         February 15, 2013
        To get started, type one of these: helpwin, helpdesk, or demo.
        For product information, visit www.mathworks.com.
        >> version()
        ans =
        8.1.0.604 (R2013a)

In this command line you are now able to run Matlab commands, load and edit scripts, but cannot display plots - they can
however be generated and exported to file, which you will need to transfer to your own machine for visualisation.
While the text mode interface is spartan, you still benefit from tab-completion (type the first few letters of
a command then press TAB twice to see possible completions) and can run the integrated help with `help command_name` 
(e.g. help plot3).
 
### Example usage of Matlab in interactive mode

At this point you should have downloaded the example scripts and started Matlab either with the graphical or the text-mode
interface. We will now test some Matlab commands by using the yahoo\_finance\_data function defined in _yahoo\_finance\_data.m_.
This function downloads stock market data through the Yahoo! Finance API, and we will use it to get 1 month worth of stock data
for IBM (whose stock symbol is 'IBM'):

         >> cd('~/matlab-tests')
         >> [hist_date, hist_high, hist_low, hist_open, hist_close, hist_vol] = yahoo_finance_data('IBM', 2014, 2, 1, 2014, 3, 1);
         >> size(hist_date)                                                                                                       
         ans =
             19     1
         >> [hist_date{1} ' ' hist_date{end}]     
         ans =
         2014-02-03 2014-02-28
         >> min(hist_low)                                                                                                         
         ans =
           171.2512
         >> max(hist_high)
         ans =
           186.1200
         >> mean(hist_close)
         ans =
           180.3184
         >> std(hist_close) 
         ans =
             4.5508

Through these commands we have seen that the function returns column vectors, we were able to get 19 days' worth of information and 
we used simple statistic functions to get an idea of how the stock varied in the given period. 

Now we will use the example1.m script that shows: 
  - how to use different plotting methods on the data retrieved with the yahoo\_finance\_data function
  - how to export the plots in different graphic formats instead of displaying them (which is only available when running the 
  full graphical environment and also allows the user to visually interact with the plot)
  
         >> example1
         Elapsed time is 2.421686 seconds.
         >> quit
         (node)$>
         (node)$> ls *pdf *eps
         example1-2dplot.eps  example1-2dplot.pdf  example1-scatter.eps

We have run the example1.m script which has downloaded Apple ('AAPL' ticker) stock data for the year 2013 and generated three plots:

  - example1-2dplot.pdf : color PDF generated with the saveas function, plotting dates (x-axis) vs closing stock price (y-axis)
  - example1-2dplot.eps : high quality black and white Enhanced PostScript (EPS) generated with the print function, same data as above
  - example1-scatter.eps : high quality color EPS generated with the print function, showing also the trading volume (z-axis) and 
using different color datapoints (red) where the closing share price was above 500

The script has also used the tic/toc Matlab commands to time it's execution and we can see it took less than 3 seconds to download
and process data from the Yahoo Finance API and generate the plots. 

Finally, we have closed our Matlab session and were returned to the cluster's command line prompt where we found the generated plots.

A PNG version of the latter two plots is shown below:
![2D Plot](https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB/plots/example1-2dplot.png)
![3D Scatter Plot](https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB/plots/example1-scatter.png)

Further examples showing serial and parallel executions are given below in the 'Example usage of Matlab in passive mode' section.
        
## Matlab execution in passive mode

For non-interactive or long executions, MATLAB can be ran in passive mode, reading all commands from
an input file you provide (e.g. named INPUTFILE.m) and saving the results in an output file (e.g. named OUTPUTFILE.out),
by either:

1. using redirection operators:

        $> matlab -nodisplay -nosplash < INPUTFILE.m > OUTPUTFILE.out

2. running the input file as a command (notice the missing '.m' extension) and copying output 
(as a log) to the output file:

        $> matlab -nodisplay -nosplash -r INPUTFILE -logfile OUTPUTFILE.out

The second usage mode is recommended as it corresponds to the batch-mode execution. In the first case your 
output file will contain the '>>' characters generated by Matlab as if ran interactively, along with the
results of your own commands.

However as the second usage mode runs your script as a command, it __must__ contain the `quit` command at
the end in order to close Matlab, otherwise after the script has executed Matlab will stay open,
waiting for further input until the end of the walltime you set for the passive job, tying up compute
resources needlessly.
        
The following minimal example shows how to run a serial (1 core) MATLAB script for 24 hours in passive mode:

        (gaia-frontend)$> oarsub -l walltime=24:00:00 "source /etc/profile; module load MATLAB; matlab -nodisplay -nosplash < INPUTFILE.m > OUTPUTFILE.out"

Ideally you __would not__ run MATLAB jobs like this but instead [create/adapt a launcher script](https://github.com/ULHPC/launcher-scripts) to contain those instructions. A minimal shell script (e.g. named 'your\_matlab\_launcher.sh') could be:

        #!/bin/bash
        source /etc/profile
        # REMEMBER to change the following to the correct paths of the input/output files:
        INPUTFILE=your_input_file_name_without_extension
        OUTPUTFILE=your_output_file_name_with_extension.out
        # Load a specific version of MATLAB and run the input script:
        module load MATLAB/2013b
        matlab -nodisplay -nosplash -r $INPUTFILE -logfile $OUTPUTFILE

then launch it in a job (e.g. requesting 6 cores on 1 node for 10 hours - assuming your input file takes advantage of the parallel cores):

        (gaia-frontend)$> oarsub -l nodes=1/core=6,walltime=10:00:00 your_matlab_launcher.sh
        
Remember! that the Matlab script you run with the '-r' parameter must contain the `quit` command at the end
in order to close Matlab properly when the script finishes.

### Example usage of Matlab in passive mode

In this section we will use the _example2.m_ script which shows:
  - the serial execution of time consuming operations; 1 core on 1 node
  - the parallel execution (based on the `parfor` command) and relative speedup vs serial execution, setting
    the maximum number of parallel threads through environment variables; up to 1 full node
  - GPU-based parallel execution; available only on [GPU-enabled nodes](https://hpc.uni.lu/systems/accelerators.html)

By default the parallel section of the script uses up to 4 threads, thus for a first test we will request 4 cores on 1
compute node for 5 minutes:
  
        (gaia-frontend)$> cd ~/matlab-tests
        (gaia-frontend)$> cat << EOF > matlab-minlauncher.sh
        #!/bin/bash
        source /etc/profile
        module load MATLAB/2013a
        matlab -nodisplay -nosplash -r example2 -logfile example2.out
        EOF
        (gaia-frontend)$> chmod +x matlab-minlauncher.sh
        (gaia-frontend)$> oarsub -l nodes=1/core=4,walltime=00:05:00 ~/matlab-tests/matlab-minlauncher.sh
        # we now wait for the job to complete execution
        (gaia-frontend)$> cat example2.out
				    < M A T L A B (R) >
			  Copyright 1984-2013 The MathWorks, Inc.
			    R2013a (8.1.0.604) 64-bit (glnxa64)
				    February 15, 2013


	To get started, type one of these: helpwin, helpdesk, or demo. 
	For product information, visit www.mathworks.com.

	-- Will perform 24 iterations on a 1000x1000 matrix

	-- Serial test
	-- Execution time: 27.870898s.
	-- Parallel tests with up to 4 cores

	-- Parallel test using 2 cores
	Starting matlabpool ... connected to 2 workers.
	Sending a stop signal to all the workers ... stopped.
	-- Execution time: 19.869666s.
	-- Execution time with overhead: 39.025023s.

	-- Parallel test using 3 cores
	Starting matlabpool ... connected to 3 workers.
	Sending a stop signal to all the workers ... stopped.
	-- Execution time: 14.584377s.
	-- Execution time with overhead: 25.587958s.

	-- Parallel test using 4 cores
	Starting matlabpool ... connected to 4 workers.
	Sending a stop signal to all the workers ... stopped.
	-- Execution time: 12.298823s.
	-- Execution time with overhead: 22.379418s.

	-- Number of processes, parallel execution time (s), parallel execution time with overhead(s), speedup, speedup with overhead:
	    1.0000   27.8709   27.8709    1.0000    1.0000
	    2.0000   19.8697   39.0250    1.4027    0.7142
	    3.0000   14.5844   25.5880    1.9110    1.0892
	    4.0000   12.2988   22.3794    2.2661    1.2454


	-- GPU-Parallel test not available on this system.


The script is also able to read an environment variable _MATLABMP_ and create as many parallel threads as specified in this variable.
We will now generate another launcher which will set this variable to the number of cores we specified to OAR.

        (gaia-frontend)$> cd ~/matlab-tests
        (gaia-frontend)$> cat << EOF > matlab-minlauncher2.sh
        #!/bin/bash
        source /etc/profile
        module load MATLAB/2013a
        export MATLABMP=$(cat $OAR_NODEFILE | wc -l)
        matlab -nodisplay -nosplash -r example2 -logfile example2.out
        EOF
        (gaia-frontend)$> chmod +x matlab-minlauncher2.sh
        (gaia-frontend)$> oarsub -l nodes=1/core=6,walltime=00:05:00 ~/matlab-tests/matlab-minlauncher2.sh
        # we now wait for the job to complete execution
        (gaia-frontend)$> head -n 17 example2.out 
				      < M A T L A B (R) >
			    Copyright 1984-2013 The MathWorks, Inc.
			      R2013a (8.1.0.604) 64-bit (glnxa64)
				      February 15, 2013

	  
	  To get started, type one of these: helpwin, helpdesk, or demo.
	  For product information, visit www.mathworks.com.
	  
	  -- Will perform 24 iterations on a 1000x1000 matrix

	  -- Serial test
	  -- Execution time: 28.193131s.

	  -- Found environment variable MATLABMP=6.
	  -- Parallel tests with up to 6 cores
        (gaia-frontend)$>  
        
We have submitted an OAR job requesting 6 cores for 5 minutes and used the second launcher. It can be seen that the example2.m script 
has read the MATLABMP environment variable and has used in its execution.

As shown previously, the jobs we have submitted did not run on GPU-enabled nodes, thus in this last example we will specifically
target GPU nodes and see that the last test of example2.m will also be executed:

      (gaia-frontend)$> cd ~/matlab-tests
      (gaia-frontend)$> oarsub -l nodes=1/core=6,walltime=00:05:00 -p "gpu='YES'" ~/matlab-tests/matlab-minlauncher2.sh
        # we now wait for the job to complete execution
      (gaia-frontend)$> tail -n 5 example2.out 
        -- GPU test 
        -- GPU Execution time: 28.192080s.
        -- GPU Execution time with overhead: 30.499892s.
        -- GPU vs Serial speedup: 1.102579.
        -- GPU with overhead vs Serial speedup: 1.019151.

The following plot shows a sample speedup obtained by using parfor on Gaia, with up to 12 parallel threads:
![Parfor speedup](https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB/plots/parfor-speedup.png)
