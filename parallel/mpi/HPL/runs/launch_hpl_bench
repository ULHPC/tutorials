#! /bin/bash
################################################################################
# launch_hpl_bench - launch HPL on the UL HPC platform 
# Time-stamp: <Thu 2013-11-07 22:52 svarrette>
#
# Copyright (c) 2013 Sebastien Varrette <Sebastien.Varrette@uni.lu>
################################################################################
#
# Submit this job in passive mode by 
#
#   oarsub [options] -S ./launcher_hpl_bench 


##########################
#                        #
#   The OAR  directives  #
#                        #
##########################
#
#          Set number of resources
#

#OAR -l enclosure=1/nodes=2,walltime=24

#          Set the name of the job (up to 15 characters,
#          no blank spaces, start with alphanumeric character)

#OAR -n HPL_ULHPCTraining

#          By default, the standard output and error streams are sent
#          to files in the current working directory with names:
#              OAR.%jobid%.stdout  <-  output stream
#              OAR.%jobid%.stderr  <-  error stream
#          where %job_id% is the job number assigned when the job is submitted.
#          Use the directives below to change the files to which the
#          standard output and error streams are sent, typically to a common file

#OAR -O HPL_ULHPCTraining-%jobid%.log
#OAR -E HPL_ULHPCTraining-%jobid%.log

#####################################
#                                   #
#   The UL HPC specific directives  #
#                                   #
#####################################

#export LC_NUMERIC=en_US.UTF

if [ -f  /etc/profile ]; then
    .  /etc/profile
fi


### Global variables
VERSION=0.1
COMMAND=`basename $0`
COMMAND_LINE="${COMMAND} $@"
VERBOSE=""
DEBUG=""
SIMULATION=""

### displayed colors
COLOR_GREEN="\033[0;32m"
COLOR_RED="\033[0;31m"
COLOR_YELLOW="\033[0;33m"
COLOR_VIOLET="\033[0;35m"
COLOR_CYAN="\033[0;36m"
COLOR_BOLD="\033[1m"
COLOR_BACK="\033[0m"

### Local variables
STARTDIR="$(pwd)"
SCRIPTFILENAME=$(basename $0)
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCH_DATADIR="${SCRIPTDIR}/data/`date +%Y-%m-%d`"

# Delay between each bench
DELAY=1

# MPI stuff
MPIRUN="mpirun"
MACHINEFILE="${OAR_NODEFILE}"
MPI_NP=1
[ -f "/proc/cpuinfo" ]   && MPI_NP=`grep processor /proc/cpuinfo | wc -l`
[ -n "${OAR_NODEFILE}" ] && MPI_NP=`cat ${OAR_NODEFILE} | wc -l`

# Number of nodes involved in the run
INVOLVED_NODES=1
[ -n "${OAR_NODEFILE}" ] && INVOLVED_NODES=`cat ${OAR_NODEFILE} | uniq | wc -l`


# HPL
HPL_SRCDIR="$HOME/TP/hpl-2.1"
HPL_ARCH="${arch}"
HPL_INFILE="HPL.dat"
HPL_OUTFILE="HPL.out"
HPL_NODES=1
HPL_Nratiostr=
HPL_Nstr="10"
HPL_N=
HPL_NBstr="1,2,3,4"
HPL_NB=
HPL_Pstr="1"
HPL_P=
HPL_Qstr="${MPI_NP}"
HPL_Q=


RAM_SIZE=16000
[ -f "/proc/meminfo" ] && RAM_SIZE=`grep MemTotal /proc/meminfo | awk '{print $2}'`

DP_ELEMENT=$(echo "sqrt(${RAM_SIZE}*1024/8)/1" | bc -l | xargs printf "%1.0f")

HPL_N1=$(echo "0.75*${DP_ELEMENT}" | bc -l | xargs printf "%1.0f")
HPL_N2=$(echo "0.8*${DP_ELEMENT}"  | bc -l | xargs printf "%1.0f")
HPL_N3=$(echo "0.85*${DP_ELEMENT}" | bc -l | xargs printf "%1.0f")
HPL_N4=$(echo "0.9*${DP_ELEMENT}"  | bc -l | xargs printf "%1.0f")
HPL_N5=$(echo "0.92*${DP_ELEMENT}" | bc -l | xargs printf "%1.0f")

HPL_Q1=${MPI_NP}
HPL_Q2=$(echo "${MPI_NP}/2" | bc -l | xargs printf "%1.0f")
HPL_Q4=$(echo "${MPI_NP}/4" | bc -l | xargs printf "%1.0f")


#######################
### Print functions ###
#######################

####
# print version of this program
##
print_version() {
    cat <<EOF
This is $COMMAND version "$VERSION".
Copyright (c) 2013 Sebastien Varrette  (http://varrette.gforge.uni.lu)
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
EOF
}

####
# print help
##
print_help() {
    less <<EOF
NAME
    $COMMAND -- Launch an HPL run on the reserve nodes

SYNOPSIS
    $COMMAND [-V | -h]
    $COMMAND [--debug] [-v] [-n]
    $COMMAND --arch ARCH 

DESCRIPTION
    $COMMAND permits to run HPL (compiled under different archs). 
    In a serious mode, the parameter is automatically computes depending on the RAM size as 
    advised (80 % of the total amount of memory).

OPTIONS
    --debug
        Debug mode. Causes $COMMAND to print debugging messages.
    -h --help
        Display a help screen and quit.
    -n --dry-run
        Simulation mode.
    -v --verbose
        Verbose mode.
    -V --version
        Display the version number then quit.
    --print 
        Print some various info
    --name NAME
        Set the name of the experiment, affects the output filename
    --dummy | --serious | --godlike
        Make a { dummy | serious | godlike } run.
         * make a dummy test to ensure your run actually work
         * in a serious mode, the problem size is set to 0.8*sqrt(n*RAM), where n is 
           the number of nodes. P is set to succesively 1,2 and 4 (Q is adapted accordingly)
           A long list of values for NB is also tested as "good" block sizes are almost always 
           in the [32 .. 256] interval so the tested values are 96,112,128,144,168,192,224,256
         * the godlike mode is similar, but test for increasing values of the problem size i.e.
           0.8*sqrt(n*RAM), 0.85*sqrt(n*RAM), 0.9*sqrt(n*RAM) and 0.92*sqrt(n*RAM)
  
    --srcdir DIR
        Set the root directory of the HPL sources (Default: ${HPL_SRCDIR}) 
    --arch ARCH 
        Set the arch (in particular, the `xhpl` run will be the one compiled in 
        SRCDIR/bin/<ARCH>
    --module MODULE
        Preload the module prior to the run.
    --hpl_{ N | NB | P | Q } NUM
        Set the value for these HPL parameter. Use a comma separated list of 
        values eventually
    --hpl_Nratio NUM
        You might want to set the ratio of the total amount of memory. 
        Use a comma separated list of values eventually
  
EXAMPLES

    Assuming you have compiled HPL in `~/src/hpl-2.1` under the arch `Linux_GCC_GotoBlas2` 
    using the OpenMPI module, run the tests as: 

       $COMMAND --module OpenMPI --arch Linux_GCC_GotoBlas2 --srcdir=$HOME/src/hpl-2.1 --dummy

    If you want to submit the same command as a passive job, use:

       oarsub -S "$COMMAND --module OpenMPI --arch Linux_GCC_GotoBlas2 --srcdir=$HOME/src/hpl-2.1 --dummy"
     
    You can also select the values explicitely:
        
       $COMMAND --hpl_N 15000 
       
       $COMMAND --hpl_N 15000,20000,40000
       
       $COMMAND --hpl_Nratio 0.8,0.9,0.92

AUTHOR
    Sebastien Varrette <Sebastien.Varrette@uni.lu>
    Web page: http://varrette.gforge.uni.lu

REPORTING BUGS
    Please report bugs to <Sebastien.Varrette@uni.lu>

COPYRIGHT
    This is free software; see the source for copying conditions.  There is
    NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.

SEE ALSO
    Other scripts are available on my web site http://varrette.gforge.uni.lu
EOF
}

######
# Print information in the following form: '[$2] $1' ($2=INFO if not submitted)
# usage: info text [title]
##
info() {
    [ -z "$1" ] && print_error_and_exit "[$FUNCNAME] missing text argument"
    local text=$1
    local title=$2
    # add default title if not submitted but don't print anything
    [ -n "$text" ] && text="${title:==>} $text"
    echo -e $text
}
debug()   { [ -n "$DEBUG"   ] && info "$1" "[${COLOR_YELLOW}DEBUG${COLOR_BACK}]"; }
verbose() { [ -n "$VERBOSE" ] && info "$1"; }
error()   { info "$1" "[${COLOR_RED}ERROR${COLOR_BACK}]"; }
warning() { info "$1" "[${COLOR_VIOLET}WARNING${COLOR_BACK}]"; }
print_error_and_exit() {
    local text=$1
    [ -z "$1" ] && text=" Bad format"
    error  "$text. '$COMMAND -h' for help."
    exit 1
}
#####
# print the strings [ OK ] or [ FAILED ] or [ FAILED ]\n$1
##
print_ok()     { echo -e "[   ${COLOR_GREEN}OK${COLOR_BACK}   ]"; }
print_failed() { echo -e "[ ${COLOR_RED}FAILED${COLOR_BACK} ]"; }
print_failed_and_exit() {
    print_failed
    [ ! -z "$1" ] && echo "$1"
    exit 1
}

#########################
### toolbox functions ###
#########################

#####
# execute a local command
# usage: execute command
###
execute() {
    [ $# -eq 0 ] && print_error_and_exit "[$FUNCNAME] missing command argument"
    debug "[$FUNCNAME] $*"
    [ -n "${SIMULATION}" ] && echo "(simulation) $*" || eval $*
    local exit_status=$?
    debug "[$FUNCNAME] exit status: $exit_status"
    return $exit_status
}

####
# ask to continue. exit 1 if the answer is no
# usage: really_continue text
##
really_continue() {
    echo -e -n "[${COLOR_VIOLET}WARNING${COLOR_BACK}] $1 Are you sure you want to continue? [Y|n] "
    read ans
    case $ans in
        n*|N*) exit 1;;
    esac
}

#####
# Check availability of binaries passed as arguments on the current system
# usage: check_bin prog1 prog2 ...
##
check_bin() {
    [ $# -eq 0 ] && print_error_and_exit "[$FUNCNAME] missing argument"
    for appl in $*; do
        echo -n -e "=> checking availability of the command '$appl' on your system \t"
        local tmp=`which $appl`
        [ -z "$tmp" ] && print_failed_and_exit "Please install $appl or check \$PATH." || print_ok
    done
}


###
# HPL Benchmark
##
bench_hpl() {
    # Prepare the input parameters

    # Eventually update the set of N if the ratio(s) of DP_ELEMENT are set
    if [ -n "${HPL_Nratiostr}" ]; then
        echo "=> select N value from the ratio ${HPL_Nratiostr} of DP_ELEMENT=${DP_ELEMENT}"
        IFS=',' read -a HPL_Nratio  <<< "${HPL_Nratiostr}"
        HPL_N=()
        for r in "${HPL_Nratio[@]}"; do
            HPL_N+=$(echo "$r*${DP_ELEMENT}" | bc -l | xargs printf "%1.0f ")
        done
    else
        IFS=',' read -a HPL_N  <<< "${HPL_Nstr}"
    fi
    IFS=',' read -a HPL_NB <<< "${HPL_NBstr}"
    IFS=',' read -a HPL_P  <<< "${HPL_Pstr}"
    IFS=',' read -a HPL_Q  <<< "${HPL_Qstr}"

    # Eventually apply the node number multiplicator
    if [ $INVOLVED_NODES -gt 1 ]; then
        echo "=> apply mutiplicator factor to each N by sqrt(${INVOLVED_NODES})"
        index=0
        while [ "$index" -lt "${#HPL_N[@]}" ]; do
            n="${HPL_N[$index]}"
            HPL_N[$index]=$(echo "sqrt(${INVOLVED_NODES})*$n" | bc -l | xargs printf "%1.0f ")
            echo "   $n updated to ${INVOLVED_NODES}*$n = ${HPL_N[$index]}"
            let "index++"
        done
    fi

    date_prefix=`date +%Hh%Mm%S`
    if [ -z "${NAME}" ]; then
        HPL_INFILE="${BENCH_DATADIR}/HPL_${date_prefix}.dat"
        HPL_OUTFILE="${BENCH_DATADIR}/`hostname`_result_HPL_${date_prefix}.out"
    else
        HPL_INFILE="${BENCH_DATADIR}/HPL_${NAME}_${date_prefix}.dat"
        HPL_OUTFILE="${BENCH_DATADIR}/`hostname`_${NAME}_HPL_${date_prefix}.out"
    fi

    debug "N  = ${HPL_N}"
    debug "NB = ${HPL_NBstr}"
    debug "P  = ${HPL_Pstr}"
    debug "Q  = ${HPL_Qstr}"
    debug "HPL_INFILE  = ${HPL_INFILE}"
    debug "HPL_OUTFILE = ${HPL_OUTFILE}"
    # exit 0

    # prepare the HPL input file
    hpl_create_inputfile HPL_N[@] HPL_NB[@] HPL_P[@] HPL_Q[@]
    [ -f ${HPL_INFILE} ] && cat ${HPL_INFILE}

    cat > ${HPL_OUTFILE} <<EOF
# ${HPL_OUTFILE}
#
# Initial command: ${COMMAND_LINE}
#
# HPL benchmark
# Generated @ `date` by:
#   ${MPI_CMD}
# (command performed in ${HPL_BENCHDIR})
### Starting timestamp: `date +%s`
EOF
    debug "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    echo "=> running '${MPI_CMD}'"
    if [ -z "${SIMULATION}" ]; then
        cd ${HPL_BENCHDIR}
        echo "   command performed in `pwd`"
        ${MPI_CMD} | tee -a ${HPL_OUTFILE}
        cd -
    fi
    echo "### Ending timestamp:   `date +%s`" >> ${HPL_OUTFILE}
    [ -f "${HPL_OUTFILE}" ] && cat ${HPL_OUTFILE}
    echo "=> now sleeping for ${DELAY}s"
    sleep $DELAY
}

# hpl_create_inputfile Narray NBarray Parray Qarray
hpl_create_inputfile() {
    [ $# -ne 4 ] && print_error_and_exit "Format: hpl_create_inputfile <N> <NB> <P> <Q>"
    declare -a Narray=("${!1}")
    declare -a NBarray=("${!2}")
    declare -a Parray=("${!3}")
    declare -a Qarray=("${!4}")
    echo "=> create input file for HPL with N=${Narray[@]}, NB=${NBarray[@]}, P=${Parray[@]} and Q=${Qarray[@]}"
    echo "   Generated file: ${HPL_INFILE}"
    cat > ${HPL_INFILE} << EOF
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
${#Narray[@]}            # of problems sizes (N)
${Narray[@]}         Ns
${#NBarray[@]}            # of NBs
${NBarray[@]}           NBs
0            PMAP process mapping (0=Row-,1=Column-major)
${#Parray[@]}            # of process grids (P x Q)
${Parray[@]}            Ps
${Qarray[@]}            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
##### This line (no. 32) is ignored (it serves as a separator). ######
0                               Number of additional problem sizes for PTRANS
1200 10000 30000                values of N
0                               number of additional blocking sizes for PTRANS
40 9 8 13 13 20 16 32 64        values of NB
EOF
    if [ -h "${HPL_BENCHDIR}/HPL.dat" ]; then
        execute "ln -sf ${HPL_INFILE} ${HPL_BENCHDIR}/HPL.dat"
    else
        # Backup eventually the previous HPL.dat
        [ -f "${HPL_BENCHDIR}/HPL.dat" ] && execute "mv ${HPL_BENCHDIR}/HPL.dat ${HPL_BENCHDIR}/HPL.dat.old"
        execute "ln -s ${HPL_INFILE} ${HPL_BENCHDIR}/HPL.dat"
    fi
    # Now there is an HPL.dat ready to be run
}




################################################################################
################################################################################
#[ $UID -gt 0 ] && print_error_and_exit "You must be root to execute this script (current uid: $UID)"

# Check for required argument
#[ $# -eq 0 ] && print_error_and_exit


# Check for options
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help)    print_help;        exit 0;;
        -V | --version) print_version;     exit 0;;
        --debug)
            DEBUG="--debug";
            VERBOSE="--verbose";;
        -v | --verbose)  VERBOSE="--verbose";;
        -n | --dry-run)  SIMULATION="--dry-run";;
        --dummy)
            HPL_Nstr="35";
            HPL_NBstr="1,2,3,4";
            HPL_Pstr="2 1 4";
            HPL_Qstr="2 4 1"
            ;;
        --serious)
            HPL_Nstr="${HPL_N2}"
            HPL_NBstr="96,112,128,144,168,192,224,256"
            HPL_Pstr="1,2,4"
            HPL_Qstr="${HPL_Q1},${HPL_Q2},${HPL_Q4}"
            ;;
        --godlike)
            HPL_Nstr="${HPL_N2},${HPL_N3},${HPL_N4},${HPL_N5}";
            HPL_NBstr="96,112,128,144,168,192,224,256";
            HPL_Pstr="1,2,4";
            HPL_Qstr="${HPL_Q1},${HPL_Q2},${HPL_Q4}"
            ;;
        --hpl_infile)  shift; HPL_INFILE="$1";;
        --hpl_outfile) shift; HPL_OUTFILE="$1";;
        --hpl_N)       shift; HPL_Nstr="$1";;
        --hpl_Nratio)  shift; HPL_Nratiostr="$1";;
        --hpl_NB)      shift; HPL_NBstr="$1";;
        --hpl_P)       shift; HPL_Pstr="$1";;
        --hpl_Q)       shift; HPL_Qstr="$1";;
        --module)      shift; MODULE_TO_LOAD="$1";;
        --mpirun)      shift; MPIRUN=$1;;
        --machinefile) shift; MACHINEFILE=$1;;
        --name)        shift; NAME=$1;;
        --srcdir)      shift; HPL_SRCDIR=$1;;
        --arch)        shift; HPL_ARCH="$1";;
        --delay)       shift; DELAY=$1;;
        --print)
            echo "N  = ${HPL_Nstr} (for 1 node)"
            echo "NB = ${HPL_NBstr}"
            echo "P  = ${HPL_Pstr}"
            echo "Q  = ${HPL_Qstr}"
            echo "HPL_INFILE  = ${HPL_INFILE}"
            echo "HPL_OUTFILE = ${HPL_OUTFILE}"
            echo "MPIRUN = ${MPIRUN}"
            echo "MPI_NP = ${MPI_NP}"
            echo "RAM_SIZE = ${RAM_SIZE}"
            echo "DP_ELEMENT = ${DP_ELEMENT}"
            echo "HPL_N1 (0,75) = ${HPL_N1}"
            echo "HPL_N2 (0,8)  = ${HPL_N2}"
            echo "HPL_N3 (0,85) = ${HPL_N3}"
            echo "HPL_N4 (0,9)  = ${HPL_N4}"
            echo "HPL_N5 (0,92) = ${HPL_N5}"
            echo "HPL_Q1 = ${HPL_Q1}"
            echo "HPL_Q2 = ${HPL_Q2}"
            echo "HPL_Q4 = ${HPL_Q4}"
            exit 0
            ;;
    esac
    shift
done

XHPL="${HPL_SRCDIR}/bin/${HPL_ARCH}/xhpl"

# Check that HPL binary exist
if [ ! -x "${XHPL}" ]; then 
    print_error_and_exit "Unable to find the xhpl binary in '${XHPL}'.\n You probably want to check the --srcdir and --arch options"
fi

if [ -n "${MODULE_TO_LOAD}" ]; then
    info "purging modules"
    execute "module purge"
    info "loading module ${MODULE_TO_LOAD}"
    execute "module load ${MODULE_TO_LOAD}"
    execute "module list"
#    `which mpirun`
fi

MPI_CMD="${MPIRUN}"

[[ "${MODULE_TO_LOAD}" =~ "OpenMPI" ]] && MPI_CMD="${MPI_CMD} -x LD_LIBRARY_PATH "
[[ "${MODULE_TO_LOAD}" =~ "MVAPICH" ]] && MPI_CMD="${MPI_CMD} -launcher ssh -launcher-exec /usr/bin/oarsh "

if [ -n "${MACHINEFILE}" -a -f "${MACHINEFILE}" ]; then
    MPI_NP=`cat ${MACHINEFILE} | wc -l`
    MPI_CMD="${MPI_CMD} -hostfile ${MACHINEFILE} -np ${MPI_NP} "
fi

MPI_CMD="${MPI_CMD} ./xhpl"

verbose "MPI command: '${MPI_CMD}'"

HPL_BENCHDIR="${HPL_SRCDIR}/bin/${HPL_ARCH}"

if [ ! -d ${BENCH_DATADIR} ]; then
    echo "=> creating ${BENCH_DATADIR}"
    execute "mkdir -p ${BENCH_DATADIR}"
fi


bench_hpl
#sleep $DELAY;
