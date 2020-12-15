#! /bin/bash
################################################################################
# setup.sh - Bootstrap Easybuild
# Time-stamp: <Tue 2020-12-15 21:42 svarrette>
################################################################################

### Local variables
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BOOTSTRAP_EB=/tmp/bootstrap_eb.py
CMD_PREFIX=

EASYBUILD_PREFIX=${EASYBUILD_PREFIX:=$HOME/.local/easybuild}
EASYBUILD_MODULES_TOOL=${EASYBUILD_MODULES_TOOL:=LMod}
EASYBUILD_MODULE_NAMING_SCHEME=${EASYBUILD_MODULE_NAMING_SCHEME:=CategorizedModuleNamingScheme}

print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
  $0 -- Boostrap EasyBuild in \$EASYBUILD_PREFIX
  Default: ${EASYBUILD_PREFIX}
    Module Tool: ${EASYBUILD_MODULES_TOOL}
    Naming Scheme: ${EASYBUILD_MODULE_NAMING_SCHEME}
USAGE
   $0 [-n]
OPTIONS
  -n --dry-run        Dry run mode
EOF
}

################################################################################
# Check for options
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help) usage; exit 0;;
        -n | --noop | --dry-run) CMD_PREFIX=echo;;
    esac
    shift
done

if [ ! -f "${BOOTSTRAP_EB}" ]; then
    echo "=> download EB bootstrap script"
    ${CMD_PREFIX} curl -o ${BOOTSTRAP_EB}  https://raw.githubusercontent.com/easybuilders/easybuild-framework/develop/easybuild/scripts/bootstrap_eb.py
fi

# Pre-check: Lmod / Environment Module exists
command -v module >/dev/null || print_error_and_exit "'module' command Not Found"

if [ -n "${EASYBUILD_PREFIX}" ]; then
    echo "=> install EasyBuild"
    ${CMD_PREFIX} python ${BOOTSTRAP_EB} $EASYBUILD_PREFIX
fi

${CMD_PREFIX} module use ${LOCAL_MODULES}
${CMD_PREFIX} module avail
${CMD_PREFIX} module load tools/EasyBuild
