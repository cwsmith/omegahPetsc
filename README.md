# omegahPetsc
omegah+petsc

The following instructions apply to the RPI AiMOS system.

## setup

Create an environment file `~/barn/envOmegahPetscGccSpectrum.sh` with the
following contents:

```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  cuda/10.2

export OMPI_CXX=g++

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/gpfs/u/home/MPFS/MPFSsmth/barn-shared/cws/software/build-omegah-dcs-gcc74-cuda/install
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/gpfs/u/home/MPFS/MPFSsmth/barn-shared/cws/software/petsc/arch-aimos-gcc74Spectrum/lib/pkgconfig
```

## build

```
source ~/barn/envOmegahPetscGccSpectrum.sh
cd ~/barn
git clone https://github.com/cwsmith/omegahPetsc.git
mkdir build-omegahPetsc
cd !$
cmake ../omegahPetsc/ -DCMAKE_CUDA_HOST_COMPILER=g++ -DCMAKE_CUDA_FLAGS="-arch=sm_70"
```
