# Usage: source settings/default.sh
#


### Align local Easybuild layout to RESIF 3
# See https://hpc-docs.uni.lu/environment/easybuild/
#
# EASYBUILD_PREFIX: [basedir]/<cluster>/<environment>/<arch>
# Ex: Default EASYBUILD_PREFIX in your home - Adapt to project directory if needed
#
_EB_PREFIX=$HOME/.local/easybuild
# ... eventually complemented with cluster
[ -n "${ULHPC_CLUSTER}" ] && _EB_PREFIX="${_EB_PREFIX}/${ULHPC_CLUSTER}"
# ... eventually complemented with software set version
_EB_PREFIX="${_EB_PREFIX}/${RESIF_VERSION_PROD}"
# ... eventually complemented with arch
[ -n "${RESIF_ARCH}" ] && _EB_PREFIX="${_EB_PREFIX}/${RESIF_ARCH}"
export EASYBUILD_PREFIX="${_EB_PREFIX}"
# Use the below variable to run:
#    module use $LOCAL_MODULES
#    module load tools/EasyBuild
export LOCAL_MODULES=${EASYBUILD_PREFIX}/modules/all

if [ -d /opt/apps/sources ]; then
    export EASYBUILD_SOURCEPATH=${EASYBUILD_PREFIX}/sources:/opt/apps/sources
fi
function mu(){
   module use $LOCAL_MODULES
   module load tools/EasyBuild
}

module use $LOCAL_MODULES
module load tools/EasyBuild 2>/dev/null && eb --show-config || echo "/!\ WARNING: Module tools/EasyBuild NOT FOUND "
