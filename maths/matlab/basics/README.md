[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/advanced/MATLAB1/MATLAB1.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/MATLAB1/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/MATLAB1/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# MATLAB (interactive, passive and sequential jobs) execution on the UL HPC platform

      Copyright (c) 2013-2018 UL HPC Team  <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/advanced/MATLAB1/cover_MATLAB1.png)](https://github.com/ULHPC/tutorials/raw/devel/advanced/MATLAB1/MATLAB1.pdf)

The objective of this tutorial is to exemplify the execution of [MATLAB](http://www.matlab.com) -
a high-level language and interactive environment for numerical computation,
visualization and programming, on top of the [UL HPC](http://hpc.uni.lu) platform.

The tutorial will show you:

1. how to run MATLAB in interactive mode, with either the full graphical interface or the text-mode interface
2. how to check the available toolboxes and licenses used
3. how to run MATLAB in passive (batch) mode, enabling unattended execution on the clusters
4. how to use MATLAB script (.m) files
5. how to plot data, saving the plots to file
6. how to take advantage of some of the paralelization capabilities of MATLAB to speed up your tasks

For the tutorial we will use the UL HPC [Gaia](http://hpc.uni.lu/systems/gaia/) cluster that includes nodes with GPU accelerators.

## Prerequisites

As part of this tutorial two Matlab example scripts have been developed and you will need to download them,
along with their dependencies, before following the instructions in the next sections:

        (gaia-frontend)$> mkdir -p ~/matlab-tutorial/code
        (gaia-frontend)$> cd ~/matlab-tutorial/code
        (gaia-frontend)$> wget --no-check-certificate https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB1/code/example1.m
        (gaia-frontend)$> wget --no-check-certificate https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB1/code/example2.m
        (gaia-frontend)$> wget --no-check-certificate https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB1/code/google_finance_data.m

Or simply clone the full tutorials repository and make a link to the MATLAB tutorial:

        (gaia-frontend)$> git clone https://github.com/ULHPC/tutorials.git
        (gaia-frontend)$> ln -s tutorials/advanced/MATLAB1/ ~/matlab-tutorial

## Matlab execution in interactive mode

### Launching the full graphical environment

Running the full MATLAB environment (e.g. on the Gaia cluster) requires an [interactive OAR session](https://hpc.uni.lu/users/docs/oar.html#concepts). When connecting to the clusters you will
need to enable X11 forwarding in order for the graphical environment to be shown on your
local machine:

- on Linux simply follow the commands below

- on OS X (depending on version) you may not have the X Window System installed,
  and thus will need to install [XQuartz](http://xquartz.macosforge.org/landing/) if
  the first command below returns an 'X11 forwarding request failed on channel 0' error

- on Windows you will need to run [VcXsrv](http://sourceforge.net/projects/vcxsrv/) first
  then to configure Putty (Connection -> SSH -> X11 -> Enable X11 forwarding) before
  logging in to the clusters.

        # Connect to Gaia with X11 forwarding enabled (Linux/OS X):
        (yourmachine)$> ssh yourlogin@access-gaia.uni.lu -p 8022 -X

        # Request an interactive job (the default parameters get you 1 core for 2 hours):
        (gaia-frontend)$> oarsub -I

        # Check the Matlab versions installed on the clusters:
        (node)$> module spider matlab

        # Load a specific MATLAB version:
        (node)$> module load base/MATLAB/2017a

        # Check that its profile has been loaded and thus we can start to use it:
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

        (yourmachine)$> ssh yourlogin@access-gaia.uni.lu -p 8022
        (gaia-frontend)$> oarsub -I
        (node)$> module load base/MATLAB/2017a

        # Launch MATLAB with the graphical display mode disabled (critical parameters):
        (node)$> matlab -nodisplay -nosplash
        Opening log file:  /home/users/vplugaru/java.log.23247

                                                                       < M A T L A B (R) >
                                                             Copyright 1984-2017 The MathWorks, Inc.
                                                              R2017a (9.2.0.538062) 64-bit (glnxa64)
                                                                        February 23, 2017
        To get started, type one of these: helpwin, helpdesk, or demo.
        For product information, visit www.mathworks.com.
        >> version()
        ans =
            '9.2.0.538062 (R2017a)

In this command line you are now able to run Matlab commands, load and edit scripts, but cannot display plots - they can
however be generated and exported to file, which you will need to transfer to your own machine for visualisation.
While the text mode interface is spartan, you still benefit from tab-completion (type the first few letters of
a command then press TAB twice to see possible completions) and can run the integrated help with `help command_name`
(e.g. help plot3).

### Example usage of Matlab in interactive mode

At this point you should have downloaded the example scripts and started Matlab either with the graphical or the text-mode
interface. We will now test some Matlab commands by using the google\_finance\_data function defined in _google\_finance\_data.m_.
This function downloads stock market data through the Google Finance API, and we will use it to get 1 month worth of stock data
for IBM (whose stock symbol is 'IBM'):

         >> cd('~/matlab-tutorial/code/')
         >> [hist_date, hist_high, hist_low, hist_open, hist_close, hist_vol] = google_finance_data('IBM', '2017-05-01', '2017-06-02');
         >> size(hist_date)
         ans =
             24     1
         >> [hist_date{1} ' ' hist_date{end}]
         ans =
             '1-May-17 2-Jun-17'
         >> min(hist_low)
         ans =
           149.7900
         >> max(hist_high)
         ans =
           160.4200
         >> mean(hist_close)
         ans =
           153.2879
         >> std(hist_close)
         ans =
             2.7618

Through these commands we have seen that the function returns column vectors, we were able to get 24 days' worth of information and
we used simple statistic functions to get an idea of how the stock varied in the given period.

Now we will use the example1.m script that shows:
  - how to use different plotting methods on the data retrieved with the google\_finance\_data function
  - how to export the plots in different graphic formats instead of displaying them (which is only available when running the
  full graphical environment and also allows the user to visually interact with the plot)

         >> example1
         Elapsed time is 1.709865 seconds.
         >> quit
         (node)$>
         (node)$> ls *pdf *eps
         example1-2dplot.eps  example1-2dplot.pdf  example1-scatter.eps

We have run the example1.m script which has downloaded Apple ('AAPL' ticker) stock data for the year 2016 and generated three plots:

  - example1-2dplot.pdf : color PDF generated with the saveas function, plotting dates (x-axis) vs closing stock price (y-axis)
  - example1-2dplot.eps : high quality black and white Enhanced PostScript (EPS) generated with the print function, same data as above
  - example1-scatter.eps : high quality color EPS generated with the print function, showing also the trading volume (z-axis) and
using different color datapoints (red) where the closing share price was above 100

The script has also used the tic/toc Matlab commands to time it's execution and we can see it took less than 2 seconds to download
and process data from the Google Finance API and generate the plots.

Finally, we have closed our Matlab session and were returned to the cluster's command line prompt where we found the generated plots.

A PNG version of the latter two plots is shown below:
![2D Plot](https://raw.githubusercontent.com/ULHPC/tutorials/devel/advanced/MATLAB1/plots/example1-2dplot.png)
![3D Scatter Plot](https://raw.githubusercontent.com/ULHPC/tutorials/devel/advanced/MATLAB1/plots/example1-scatter.png)

Further examples showing serial and parallel executions are given below in the 'Example usage of Matlab in passive mode' section.

## Checking available toolboxes and license status

In order to be able to run MATLAB and specific features provided through the various MATLAB toolboxes, sufficient licenses need to
be available. The state of the licenses can be checked with the `lmstat` utility.

First, we will check that the license server is running (an __UP__ status should be shown in the output of lmutil):

         (node)$> module load base/MATLAB
         (node)$> $EBROOTMATLAB/etc/glnxa64/lmutil lmstat -c $EBROOTMATLAB/licenses/network.lic

Next, we can check the total number of MATLAB licenses available (issued) and how many are used:

         (node)$> $EBROOTMATLAB/etc/glnxa64/lmutil lmstat -c $EBROOTMATLAB/licenses/network.lic -f MATLAB

To check for a specific feature and its usage (e.g. the financial toolbox if we know its name):

         (node)$> $EBROOTMATLAB/etc/glnxa64/lmutil lmstat -c $EBROOTMATLAB/licenses/network.lic -f Financial_toolbox

To see all available toolboxes:

         (node)$> $EBROOTMATLAB/etc/glnxa64/lmutil lmstat -c $EBROOTMATLAB/licenses/network.lic -a

Checking the availability of statistics toolboxes (if we don't know the exact name, but that 'stat' is in the name):

         (node)$> $EBROOTMATLAB/etc/glnxa64/lmutil lmstat -c $EBROOTMATLAB/licenses/network.lic -a | grep -i stat

Finally, checking the available toolboxes (but with no information on the specific # of available/used licenses), can be done directly from MATLAB, e.g.:

         (node)$> module load base/MATLAB/2017a
         (node)$> matlab -nodesktop -nodisplay
         Opening log file:  /home/users/vplugaru/java.log.6343
                                                                     < M A T L A B (R) >
                                                           Copyright 1984-2017 The MathWorks, Inc.
                                                            R2017a (9.2.0.538062) 64-bit (glnxa64)
                                                                      February 23, 2017


         To get started, type one of these: helpwin, helpdesk, or demo.
         For product information, visit www.mathworks.com.

         >> ver
         ----------------------------------------------------------------------------------------------------
         MATLAB Version: 9.2.0.538062 (R2017a)
         MATLAB License Number: 886910
         Operating System: Linux 3.16.0-4-amd64 #1 SMP Debian 3.16.43-2+deb8u3 (2017-08-15) x86_64
         Java Version: Java 1.7.0_60-b19 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode
         ----------------------------------------------------------------------------------------------------
         MATLAB                                                Version 9.2         (R2017a)
         Simulink                                              Version 8.9         (R2017a)
         Aerospace Blockset                                    Version 3.19        (R2017a)
         Aerospace Toolbox                                     Version 2.19        (R2017a)
         Antenna Toolbox                                       Version 2.2         (R2017a)
         Bioinformatics Toolbox                                Version 4.8         (R2017a)
         Communications System Toolbox                         Version 6.4         (R2017a)
         Computer Vision System Toolbox                        Version 7.3         (R2017a)
         Control System Toolbox                                Version 10.2        (R2017a)
         Curve Fitting Toolbox                                 Version 3.5.5       (R2017a)
         DSP System Toolbox                                    Version 9.4         (R2017a)
         Database Toolbox                                      Version 7.1         (R2017a)
         Datafeed Toolbox                                      Version 5.5         (R2017a)
         Econometrics Toolbox                                  Version 4.0         (R2017a)
         Embedded Coder                                        Version 6.12        (R2017a)
         Filter Design HDL Coder                               Version 3.1.1       (R2017a)
         Financial Instruments Toolbox                         Version 2.5         (R2017a)
         Financial Toolbox                                     Version 5.9         (R2017a)
         Fixed-Point Designer                                  Version 5.4         (R2017a)
         Fuzzy Logic Toolbox                                   Version 2.2.25      (R2017a)
         Global Optimization Toolbox                           Version 3.4.2       (R2017a)
         HDL Coder                                             Version 3.10        (R2017a)
         HDL Verifier                                          Version 5.2         (R2017a)
         Image Acquisition Toolbox                             Version 5.2         (R2017a)
         Image Processing Toolbox                              Version 10.0        (R2017a)
         Instrument Control Toolbox                            Version 3.11        (R2017a)
         LTE System Toolbox                                    Version 2.4         (R2017a)
         MATLAB Coder                                          Version 3.3         (R2017a)
         MATLAB Compiler                                       Version 6.4         (R2017a)
         MATLAB Compiler SDK                                   Version 6.3.1       (R2017a)
         MATLAB Report Generator                               Version 5.2         (R2017a)
         Mapping Toolbox                                       Version 4.5         (R2017a)
         Model Predictive Control Toolbox                      Version 5.2.2       (R2017a)
         Neural Network Toolbox                                Version 10.0        (R2017a)
         Optimization Toolbox                                  Version 7.6         (R2017a)
         Parallel Computing Toolbox                            Version 6.10        (R2017a)
         Partial Differential Equation Toolbox                 Version 2.4         (R2017a)
         Phased Array System Toolbox                           Version 3.4         (R2017a)
         RF Blockset                                           Version 6.0         (R2017a)
         RF Blockset                                           Version 6.0         (R2017a)
         RF Toolbox                                            Version 3.2         (R2017a)
         Robotics System Toolbox                               Version 1.4         (R2017a)
         Robust Control Toolbox                                Version 6.3         (R2017a)
         Signal Processing Toolbox                             Version 7.4         (R2017a)
         SimBiology                                            Version 5.6         (R2017a)
         SimEvents                                             Version 5.2         (R2017a)
         Simscape                                              Version 4.2         (R2017a)
         Simscape Driveline                                    Version 2.12        (R2017a)
         Simscape Electronics                                  Version 2.11        (R2017a)
         Simscape Fluids                                       Version 2.2         (R2017a)
         Simscape Multibody                                    Version 5.0         (R2017a)
         Simscape Power Systems                                Version 6.7         (R2017a)
         Simulink 3D Animation                                 Version 7.7         (R2017a)
         Simulink Code Inspector                               Version 3.0         (R2017a)
         Simulink Coder                                        Version 8.12        (R2017a)
         Simulink Control Design                               Version 4.5         (R2017a)
         Simulink Design Optimization                          Version 3.2         (R2017a)
         Simulink Design Verifier                              Version 3.3         (R2017a)
         Simulink Report Generator                             Version 5.2         (R2017a)
         Simulink Test                                         Version 2.2         (R2017a)
         Simulink Verification and Validation                  Version 3.13        (R2017a)
         Stateflow                                             Version 8.9         (R2017a)
         Statistics and Machine Learning Toolbox               Version 11.1        (R2017a)
         Symbolic Math Toolbox                                 Version 7.2         (R2017a)
         System Identification Toolbox                         Version 9.6         (R2017a)
         Trading Toolbox                                       Version 3.2         (R2017a)
         Vision HDL Toolbox                                    Version 1.4         (R2017a)
         Wavelet Toolbox                                       Version 4.18        (R2017a)

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

        (gaia-frontend)$> oarsub -l walltime=24:00:00 "source /etc/profile; module load base/MATLAB; matlab -nodisplay -nosplash < INPUTFILE.m > OUTPUTFILE.out"

Ideally you __would not__ run MATLAB jobs like this but instead [create/adapt a launcher script](https://github.com/ULHPC/launcher-scripts) to contain those instructions. A minimal shell script (e.g. named 'your\_matlab\_launcher.sh') could be:

        #!/bin/bash
        source /etc/profile
        # REMEMBER to change the following to the correct paths of the input/output files:
        INPUTFILE=your_input_file_name_without_extension
        OUTPUTFILE=your_output_file_name_with_extension.out
        # Load a specific version of MATLAB and run the input script:
        module load base/MATLAB/2017a
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

By default the parallel section of the script uses up to 4 threads, thus for a first test we will:

* create a _launcher script_ called matlab-minlauncher.sh (paste in your terminal from `cat` to `EOF` - [see here how this is done](http://tldp.org/LDP/abs/html/here-docs.html),
or create the file yourself with the respective content)
* make it executable by setting the corresponding filesystem permission ([see here for more details](https://en.wikipedia.org/wiki/Modes_(Unix)))
* request from the OAR scheduler 4 cores on 1 compute node for 5 minutes
* wait until the job completes its execution (see its status with `oarstat -j $JOBID` and full details with `oarstat -f -j $JOBID`):

      (gaia-frontend)$> cd ~/matlab-tutorial/code
      (gaia-frontend)$> cat << EOF > matlab-minlauncher.sh
      #!/bin/bash
      source /etc/profile
      module load base/MATLAB/2017a
      cd ~/matlab-tutorial/code
      matlab -nodisplay -nosplash -r example2 -logfile example2.out
      EOF
      (gaia-frontend)$> chmod +x matlab-minlauncher.sh
      (gaia-frontend)$> oarsub -l nodes=1/core=4,walltime=00:05:00 ~/matlab-tutorial/code/matlab-minlauncher.sh
      (gaia-frontend)$> cat example2.out
                             < M A T L A B (R) >
                   Copyright 1984-2017 The MathWorks, Inc.
                    R2017a (9.2.0.538062) 64-bit (glnxa64)
                              February 23, 2017

      To get started, type one of these: helpwin, helpdesk, or demo.
      For product information, visit www.mathworks.com.

      -- Will perform 24 iterations on a 1000x1000 matrix

      -- Serial test
      -- Execution time: 15.143097s.
      -- Parallel tests with up to 4 cores

      -- Parallel test using 2 cores
      Starting parallel pool (parpool) using the 'local' profile ...
      connected to 2 workers.
      Parallel pool using the 'local' profile is shutting down.
      -- Execution time: 10.588108s.
      -- Execution time with overhead: 51.693702s.

      -- Parallel test using 3 cores
      Starting parallel pool (parpool) using the 'local' profile ...
      connected to 3 workers.
      Parallel pool using the 'local' profile is shutting down.
      -- Execution time: 6.646897s.
      -- Execution time with overhead: 21.025083s.

      -- Parallel test using 4 cores
      Starting parallel pool (parpool) using the 'local' profile ...
      connected to 4 workers.
      Parallel pool using the 'local' profile is shutting down.
      -- Execution time: 5.196755s.
      -- Execution time with overhead: 19.772026s.

      -- Number of processes, parallel execution time (s), parallel execution time with overhead(s), speedup, speedup with overhead:
        1.0000   15.1431   15.1431    1.0000    1.0000
        2.0000   10.5881   51.6937    1.4302    0.2929
        3.0000    6.6469   21.0251    2.2782    0.7202
        4.0000    5.1968   19.7720    2.9140    0.7659


      -- GPU-Parallel test not available on this system.


The next launcher script is also able to read an environment variable _MATLABMP_ and create as many parallel threads as specified in this variable.
We will now generate another launcher which will set this variable to the number of cores we specified to OAR.

    (gaia-frontend)$> cd ~/matlab-tutorial/code
    (gaia-frontend)$> cat << EOF > matlab-minlauncher2.sh
    #!/bin/bash
    source /etc/profile
    module load base/MATLAB/2017a
    cd ~/matlab-tutorial/code
    export MATLABMP=\$(cat $OAR_NODEFILE | wc -l)
    matlab -nodisplay -nosplash -r example2 -logfile example2b.out
    EOF
    (gaia-frontend)$> chmod +x matlab-minlauncher2.sh
    (gaia-frontend)$> oarsub -l nodes=1/core=6,walltime=00:05:00 ~/matlab-tutorial/code/matlab-minlauncher2.sh
    # we now wait for the job to complete execution - check its progress yourself with oarstat
    (gaia-frontend)$> head -n 17 example2b.out
                           < M A T L A B (R) >
                 Copyright 1984-2017 The MathWorks, Inc.
                  R2017a (9.2.0.538062) 64-bit (glnxa64)
                            February 23, 2017

    To get started, type one of these: helpwin, helpdesk, or demo.
    For product information, visit www.mathworks.com.

    -- Will perform 24 iterations on a 1000x1000 matrix

    -- Serial test
    -- Execution time: 15.227198s.

    -- Found environment variable MATLABMP=6.
    -- Parallel tests with up to 6 cores
    (gaia-frontend)$>

We have submitted an OAR job requesting 6 cores for 5 minutes and used the second launcher. It can be seen that the example2.m script has read the MATLABMP environment variable and has used in its execution. Take a look at the MATLAB code to see how this is accomplished.

As shown previously, the jobs we have submitted did not run on GPU-enabled nodes, thus in this last example we will specifically target GPU nodes and see that the last test of example2.m will also be executed.
Before testing the following commands, edit the `matlab-minlauncher2.sh` script and make MATLAB store its output in a `example2c.out`
file.

      (gaia-frontend)$> cd ~/matlab-tutorial/code
      (gaia-frontend)$> oarsub -l nodes=1/core=6,walltime=00:05:00 -p "gpu='YES'" ~/matlab-tutorial/code/matlab-minlauncher2.sh
      # now wait for the job to complete execution, then check the output file
      (gaia-frontend)$> tail -n 5 example2c.out
      -- GPU test
      -- GPU Execution time: 19.809493s.
      -- GPU Execution time with overhead: 21.399610s.
      -- GPU vs Serial speedup: 1.249748.
      -- GPU with overhead vs Serial speedup: 1.156884.

The following plot shows a sample speedup obtained by using parfor on Gaia, with up to 12 parallel threads:
![Parfor speedup](https://raw.github.com/ULHPC/tutorials/devel/advanced/MATLAB1/plots/parfor-speedup.png)

Relative to the fast execution of the inner instruction (which calculates the eigenvalues of a matrix)
the overhead given by the creation of the parallel pool and the task assignations is quite high in this example,
where for 12 cores the speedup is 5.26x but taking the overhead into account it is only 4x.

## Useful references

  - [Getting Started with Parallel Computing Toolbox](http://nl.mathworks.com/help/distcomp/getting-started-with-parallel-computing-toolbox.html)
  - [Parallel for-Loops (parfor) documentation](https://nl.mathworks.com/help/distcomp/parfor.html)
  - [GPU Computing documentation](https://nl.mathworks.com/discovery/matlab-gpu.html)
