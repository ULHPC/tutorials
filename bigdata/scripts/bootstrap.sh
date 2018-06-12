#! /bin/bash -l
#            ^^
#         Mandatory for loading the modules
################################################################################
# bootstrap.sh - Bootstrap Java, eventually with Easybuild
# Time-stamp: <Tue 2018-06-12 22:14 svarrette>
################################################################################

DO_EASYBUILD=
INSTALL_JAVA7=
INSTALL_JAVA8=
INSTALL_MAVEN=

OUTPUT_DIR=.

### Java 7u80
#ARCHIVE_JAVA7_URL='http://download.oracle.com/otn-pub/java/jdk/7u80-b15/jdk-7u80-linux-x64.tar.gz'
ARCHIVE_JAVA7_URL='http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-7u80-linux-x64.tar.gz'
ARCHIVE_JAVA7=$(basename "${ARCHIVE_JAVA7_URL}")
JAVA7_EB='Java-1.7.0_80.eb'


### Java 8u152
ARCHIVE_JAVA8_URL='http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz'
ARCHIVE_JAVA8=$(basename "${ARCHIVE_JAVA8_URL}")
JAVA8_EB='Java-1.8.0_152.eb'

# Latest Maven
MAVEN_EB='Maven-3.5.2.eb'

# Hadoop
HADOOP_EB='Hadoop-2.6.0-cdh5.12.0-native.eb'


if [ -f  /etc/profile ]; then
   .  /etc/profile
fi

###
# Usage: eb_install <filename>.eb
##
function eb_install() {
    local ebconfig=$1
    [ -z "$ebconfig" ] && return || true
    cmd="cd ${OUTPUT_DIR} && eb $ebconfig --robot-paths=$PWD/${OUTPUT_DIR}: "
    echo "==> running '${cmd} -D'"
    ${cmd} -D
    echo "==> running '${cmd}'"
    ${cmd}
}

##########################################################
# Check for options
while [ $# -ge 1 ]; do
    case $1 in
        -e  | -eb | --eb | --easybuild) DO_EASYBUILD=$1;;
        -a  | --all)
            INSTALL_JAVA7=$1
            INSTALL_JAVA8=$1
            INSTALL_MAVEN=$1
            ;;
        -d  | --output-dir)       shift; OUTPUT_DIR=$1;;
        -7  | -java7 | --java7)   INSTALL_JAVA7=$1;;
        -8  | -java8 | --java8)   INSTALL_JAVA8=$1;;
        -m  | --maven)            INSTALL_MAVEN=$1;;
        -ha | -hadoop | --hadoop) INSTALL_HADOOP=$1;;
    esac
    shift
done

if [ -n "${DO_EASYBUILD}" ]; then
    # mu
    module use $LOCAL_MODULES
    module load tools/EasyBuild
fi

# Download and eventually install Java 7
if [ -n "${INSTALL_JAVA7}" ]; then
    if [ ! -f "${OUTPUT_DIR}/${ARCHIVE_JAVA7}" ]; then
        echo "==> Downloading Java 7 archive '${ARCHIVE_JAVA7}'"
        # curl -OL -H "Cookie: oraclelicense=accept-securebackup-cookie" "${ARCHIVE_JAVA7_URL}"
        wget ${ARCHIVE_JAVA7_URL}
        if [ "${OUTPUT_DIR}" != '.' ]; then
            mv ${ARCHIVE_JAVA7} ${OUTPUT_DIR}/
        fi
    fi
    if [ -n "${DO_EASYBUILD}" ]; then
        cd ${OUTPUT_DIR} && eb "${JAVA7_EB}" -Dr
        cd ${OUTPUT_DIR} && eb "${JAVA7_EB}" -r
    fi
fi

# Download and eventually install Java 8
if [ -n "${INSTALL_JAVA8}" ]; then
    if [ ! -f "${OUTPUT_DIR}/${ARCHIVE_JAVA8}" ]; then
        echo "==> Downloading Java 8 archive '${ARCHIVE_JAVA8}'"
        # curl -OL -H "Cookie: oraclelicense=accept-securebackup-cookie" "${ARCHIVE_JAVA8_URL}"
        wget ${ARCHIVE_JAVA8_URL}
        if [ "${OUTPUT_DIR}" != '.' ]; then
            mv ${ARCHIVE_JAVA8} ${OUTPUT_DIR}/
        fi
    fi
    if [ -n "${DO_EASYBUILD}" ]; then
        cd ${OUTPUT_DIR} && eb "${JAVA8_EB}" -Dr
        cd ${OUTPUT_DIR} && eb "${JAVA8_EB}" -r
    fi
fi

# eventually install Maven
if [ -n "${INSTALL_MAVEN}" ] && [ -n "${DO_EASYBUILD}" ]; then
    eb_install "${MAVEN_EB}"
fi

# eventually install Hadoop
if [ -n "${INSTALL_HADOOP}" ] && [ -n "${DO_EASYBUILD}" ] && [ -f "${HADOOP_EB}" ]; then
    module load devel/Maven/3.5.2 devel/protobuf/2.5.0 devel/CMake/3.9.1 lib/snappy/1.1.6
    eb_install "${HADOOP_EB}"
fi
