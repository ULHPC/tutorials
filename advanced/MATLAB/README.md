`README.md`

Copyright (c) 2014 Valentin Plugaru <Valentin.Plugaru@gmail.com>

-------------------


# UL HPC Tutorial: MATLAB execution on the UL HPC platform

The objective of this tutorial is to show the execution of [Matlab](http://www.matlab.com) - 
a high-level language and interactive environment for numerical computation, 
visualization and programming, on top of the [UL HPC](http://hpc.uni.lu) platform.

## Prerequisites

As part of this tutorial two example scripts have been developed and you will need to download them,
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

 In this command line you are now able to run commands, load and edit scripts, but cannot display plots - they can
 however be generated and exported to file, which you will need to transfer to your own machine for visualisation.
 While the text mode interface is spartan, you still benefit from tab-completion (type the first few letters of
 a command then press TAB TAB) and can run the integrated help with `help command_name` (e.g. help plot3).
        
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
the end in order to exit the environment, otherwise after the script has executed Matlab will stay open,
waiting for further input until the end of the walltime you set for the passive job.
        
The following minimal example shows how to run a serial (1 core) MATLAB script for 24 hours in passive mode:

        (gaia-frontend)$> oarsub -l walltime=24:00:00 "source /etc/profile; module load MATLAB; matlab -nodisplay -nosplash < INPUTFILE.m > OUTPUTFILE.out"

Ideally you __would not__ run MATLAB jobs like this but instead [create/adapt a launcher script](https://github.com/ULHPC/launcher-scripts) to contain those instructions. A minimal shell script (e.g. named 'your\_matlab\_launcher.sh') could be:

        #!/bin/bash
        source /etc/profile
        # REMEMBER to change the following to the correct paths of the input/output files:
        INPUTFILE=your_input_file_name_without_extension
        OUTPUTFULE=your_output_file_name_with_extension.out
        # Load a specific version of MATLAB and run the input script:
        module load MATLAB/2013b
        matlab -nodisplay -nosplash -r $INPUTFILE -logfile $OUTPUTFILE

then launch it in a job (e.g. requesting 6 cores on 1 node for 10 hours - assuming your input file takes advantage of the parallel cores):

        (gaia-frontend)$> oarsub -l nodes=1/core=6,walltime=10:00:00 your_matlab_launcher.sh
        
Remember! that the Matlab script you run with the '-r' parameter must contain the `quit` command at the end
in order to close Matlab properly when the script finishes.