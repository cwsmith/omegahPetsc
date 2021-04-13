# omegahPetsc
omegah+petsc

The following instructions apply to the RPI AiMOS system.

## install petsc

We will be working in the `barn` directory.  If you are curious, you can read
more about the filesystem here:
https://secure.cci.rpi.edu/wiki/index.php?title=File_System

### clone the PETSc repo

```
cd ~/barn
git clone https://gitlab.com/petsc/petsc.git
```

The clone command will create a `petsc` directory and produce the following
output:

```
Cloning into 'petsc'...
remote: Enumerating objects: 878765, done.
remote: Counting objects: 100% (878765/878765), done.
remote: Compressing objects: 100% (202874/202874), done.
remote: Total 878765 (delta 673316), reused 876417 (delta 671285), pack-reused 0
Receiving objects: 100% (878765/878765), 211.03 MiB | 26.44 MiB/s, done.
Resolving deltas: 100% (673316/673316), done.
Checking out files: 100% (8528/8528), done.
```

### create an environment script

Enter the `petsc` directory:

```
cd petsc
```

and paste the following commands into a file named `envGnu74Cuda.sh`:

```
module use
/gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  cuda/10.2 \
  netlib-lapack \
  hdf5 \
  parmetis/4.0.3-int32-real32-nreyjmh \
  hypre/2.18.1-int32-rgc66ne

export OMPI_CXX=g++
export OMPI_CC=gcc
export OMPI_FC=gfortran
```

These commands setup the environment for building PETSc with the GNU compilers.

### create a configure script

Paste the following commands into a file named `arch-aimos.py`:

```
#!/usr/bin/python
if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=mpicc',
    '--with-cxx=mpiCC',
    '--with-fc=mpif90',
    '--with-shared-libraries=1',
    '--with-debugging=yes',
    '--COPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--CXXOPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--FOPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--with-blaslapack-lib=-L' + os.environ['NETLIB_LAPACK_ROOT'] + '/lib64
-lblas -llapack',
    '--with-cuda=1',
    '--with-cudac=nvcc',
    '--with-parmetis-dir=' + os.environ['PARMETIS_ROOT'],
    '--with-metis-dir=' + os.environ['METIS_ROOT'],
    '--with-make-np=16',
  ] 
  configure.petsc_configure(configure_options)
```

Make the script executable:

```
chmod +x arch-aimos.py
```

### configure the PETSc build

'Source' the environment script to export the settings within it to your active
terminal

```
source envGnu74Cuda.sh
```

Run the configure script:

```
./arch-aimos.py
```

This will take ~six minutes and will produce the following output if it
succeeded:

```
xxx=========================================================================xxx
 Configure stage complete. Now build PETSc libraries with:
   make PETSC_DIR=/gpfs/u/barn/MPFS/MPFSzhqg/petsc PETSC_ARCH=arch-aimos all
xxx=========================================================================xxx
```

### run make

As instructed by configure, we will now run make to compile the petsc libraries. 

```
make PETSC_DIR=/gpfs/u/barn/MPFS/MPFSzhqg/petsc PETSC_ARCH=arch-aimos all
```

This will take several minutes.  If it succeeds the following output will
appear:

```
Now to check if the libraries are working do:
make PETSC_DIR=/gpfs/u/barn/MPFS/MPFSzhqg/petsc PETSC_ARCH=arch-aimos check
=========================================
```

To run the 'make check' command PETSc provided we will need to allocate a
compute node.

## install omega_h

### download

```
cd ~/barn
git clone https://github.com/SNLComputation/omega_h.git
```

### setup

Create an environment script `~/barn/omega_h/envOmegaAimosGcc.sh` with the
following contents:

```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  cuda/10.2

export OMPI_CXX=g++
```


### build

```
source ~/barn/omega_h/envOmegaAimosGcc.sh 
mkdir ~/barn/build-omegah-AimosGcc74
cd !$
cmake ~/barn/omega_h \
-DCMAKE_INSTALL_PREFIX=$PWD/install \
-DBUILD_SHARED_LIBS=OFF \
-DOmega_h_USE_CUDA=on \
-DOmega_h_USE_MPI=on \
-DCMAKE_CUDA_HOST_COMPILER=`which mpicxx` \
-DCMAKE_CUDA_FLAGS="-arch=sm_70" 

make install -j16 
```

## install omegahPetsc

### download

```
cd ~/barn
git clone https://github.com/cwsmith/omegahPetsc.git
```

### setup

Create an environment file `~/barn/omegahPetsc/envOmegahPetscGccSpectrum.sh` with the
following contents:

```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  cuda/10.2

export OMPI_CXX=g++

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:~/barn/build-omegah-AimosGcc74/install/lib/cmake
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:~/barn/petsc/arch-aimos/lib/pkgconfig
```

## build

```
source ~/barn/omegahPetsc/envOmegahPetscGccSpectrum.sh
mkdir build-omegahPetsc
cd !$
cmake ../omegahPetsc/ -DOMEGAH_PETSC_USE_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=mpicxx -DCMAKE_CUDA_FLAGS="-arch=sm_70"
make
```


# Alternative instructions for a build of petsc and omegah with cuda disabled for easier debugging

## petsc


the following script, `envGnu74CudaOff.sh`, substitutes `envGnu74Cuda.sh` from the parallel build
arch
```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  netlib-lapack \
  hdf5 \
  parmetis/4.0.3-int32-real32-nreyjmh \
  hypre/2.18.1-int32-rgc66ne

export OMPI_CXX=g++
export OMPI_CC=gcc
export OMPI_FC=gfortran
```

the following script, `arch-aimos-cudaOff.py`, substitutes for `arch-aimos.py`

```
#!/usr/bin/python
if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=mpicc',
    '--with-cxx=mpiCC',
    '--with-fc=mpif90',
    '--with-shared-libraries=1',
    '--with-debugging=yes',
    '--COPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--CXXOPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--FOPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--with-blaslapack-lib=-L' + os.environ['NETLIB_LAPACK_ROOT'] + '/lib64 -lblas -llapack',
    '--with-parmetis-dir=' + os.environ['PARMETIS_ROOT'],
    '--with-metis-dir=' + os.environ['METIS_ROOT'],
    '--with-make-np=16',
  ]
  configure.petsc_configure(configure_options)
```

## omegah

the following script, `envOmegaCudaOffAimosGcc.sh`, substitutes for `~/barn/omega_h/envOmegaAimosGcc.sh`

```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load cmake/3.15.4-mnqjvz6

export OMPI_CXX=g++
```

the following `cmake` command should be ran instead of the one listed in the parallel section

```
source ~/barn/omega_h/envOmegaCudaOffAimosGcc.sh 
mkdir ~/barn/build-omegahCudaOff-AimosGcc74
cd !$
cmake ~/barn/omega_h \
-DCMAKE_INSTALL_PREFIX=$PWD/install \
-DBUILD_SHARED_LIBS=OFF \
-DOmega_h_USE_CUDA=off \
-DOmega_h_USE_MPI=on \
-DCMAKE_CXX_COMPILER=`which mpicxx`

make install -j16 
```

## omegahPetsc

The `parallel_box_test_nocuda` branch is required to run without cuda:

https://github.com/cwsmith/omegahPetsc/commit/1d339dfd3ffb8b3c90fc699a1e91883a6bf66696

Run the following command to switch branches:

```
git checkout parallel_box_test_nocuda
```

the following script, `envOmegahCudaOffPetscGccSpectrum.sh`, substitutes for `~/barn/omegahPetsc/envOmegahPetscGccSpectrum.sh`

```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load cmake/3.15.4-mnqjvz6

export OMPI_CXX=g++

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:~/barn/build-omegahCudaOff-AimosGcc74/install/lib/cmake
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:~/barn/petsc/arch-aimos-cudaOff/lib/pkgconfig
```

doConfigCudaOff.sh

```
source ~/barn/omegahPetsc/envOmegahCudaOffPetscGccSpectrum.sh
mkdir build-omegahPetsc-cudaOff
cd !$
cmake ../omegahPetsc/ -DCMAKE_CXX_COMPILER=mpicxx
make
```

# Build on SCOREC RHEL7 with cuda

The instructions download and install packages into the
`~/develop/omegahPetscCuda` directory.  The variable `root`
is set to this path.  Ensure that this is set before running
any of the instructions below.

```
export root=~/develop/omegahPetscCuda
```

## one time setup

```
mkdir -p $root
```

## petsc

```
cd $root
git clone https://gitlab.com/petsc/petsc.git
cd petsc
# some PETSc APIs appear to have changed in main@4982446
git checkout 7a013d0
```

environment script `$root/petsc/envCuda.sh`

```
module use /opt/scorec/spack/v0132/lmod/linux-rhel7-x86_64/Core
module load gcc mpich 
module load \
  cmake \
  hdf5 \
  parmetis/4.0.3-int32-uuza7iv \
  hypre/2.18.1-int32-y2p4vsy \
  cuda/10.2

export MPICH_CXX=g++
export MPICH_CC=gcc
export MPICH_FC=gfortran
```

configure script `$root/petsc/arch-rhel7-cuda.py`

```
#!/usr/bin/python
if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=mpicc',
    '--with-cxx=mpicxx',
    '--with-fc=mpif90',
    '--with-shared-libraries=1',
    '--with-debugging=yes',
    '--COPTFLAGS=-g -O2 -fPIC',
    '--CXXOPTFLAGS=-g -O2 -fPIC',
    '--FOPTFLAGS=-g -O2 -fPIC',
    '--with-parmetis-dir=' + os.environ['PARMETIS_ROOT'],
    '--with-metis-dir=' + os.environ['METIS_ROOT'],
    '--with-make-np=8',
    '--with-cuda=1',
    '--with-cudac=nvcc',
  ] 
  configure.petsc_configure(configure_options)
```

build

```
chmod +x arch-rhel7-cuda.py
source envCuda.sh
./arch-rhel7-cuda.py
#follow the instructions in the petsc configure output to run make
```

## omegah

```
cd $root
git clone https://github.com/SCOREC/omega_h.git
```


build

```
#source the petsc environment script if not done so already
mkdir $root/build-omegahCudaOn-rhel7
cd $_
cmake $root/omega_h \
-DCMAKE_INSTALL_PREFIX=$PWD/install \
-DOmega_h_USE_MPI=on \
-DOmega_h_USE_CUDA=on \
-DCMAKE_CUDA_HOST_COMPILER=g++ \
-DCMAKE_CUDA_FLAGS="-arch=sm_75" \
-DMPI_CXX_COMPILER=`which mpicxx` \
-DBUILD_TESTING=ON

make install -j16
```

## omegahPetsc

```
cd $root
git clone https://github.com/cwsmith/omegahPetsc.git
```

environment script `$root/omegahPetsc/envRhel7v0132_omegahPetsc_cuda.sh`

```
module use /opt/scorec/spack/v0132/lmod/linux-rhel7-x86_64/Core
module load gcc mpich cmake cuda/10.2
export MPICH_CXX=g++

oh=$root/build-omegahCudaOn-rhel7
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$oh/install/lib/cmake
petscArch=arch-rhel7-cuda
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$root/petsc/${petscArch}/lib/pkgconfig
```

build

```
source $root/omegahPetsc/envRhel7v0132_omegahPetsc_cuda.sh
mkdir $root/build-omegahPetsc-cudaOn
cd $_
cmake ../omegahPetsc/ \
 -DOMEGAH_PETSC_USE_CUDA=ON \
 -DCMAKE_CUDA_HOST_COMPILER=mpicxx \
 -DCMAKE_CUDA_ARCHITECTURES="75"
make
```

# Build on SCOREC RHEL7 with cuda disabled

## petsc

```
mkdir ~/develop
cd ~/develop
git clone https://gitlab.com/petsc/petsc.git
cd petsc
git checkout master
```

environment script `~/develop/petsc/envNoCuda.sh`

```
module use /opt/scorec/spack/v0132/lmod/linux-rhel7-x86_64/Core
module load gcc mpich 
module load \
  cmake \
  hdf5 \
  parmetis/4.0.3-int32-uuza7iv \
  hypre/2.18.1-int32-y2p4vsy
export MPICH_CXX=g++
export MPICH_CC=gcc
export MPICH_FC=gfortran
```

configure script `~/develop/petsc/arch-rhel7-nocuda.py`

```
#!/usr/bin/python
if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=mpicc',
    '--with-cxx=mpicxx',
    '--with-fc=mpif90',
    '--with-shared-libraries=1',
    '--with-debugging=yes',
    '--COPTFLAGS=-g -O2 -fPIC',
    '--CXXOPTFLAGS=-g -O2 -fPIC',
    '--FOPTFLAGS=-g -O2 -fPIC',
    '--with-parmetis-dir=' + os.environ['PARMETIS_ROOT'],
    '--with-metis-dir=' + os.environ['METIS_ROOT'],
    '--with-make-np=8',
  ] 
  configure.petsc_configure(configure_options)
```

build

```
source envNoCuda.sh
./arch-rhe7-nocuda.py
#follow the instructions in the petsc configure output to run make
```


## omegah

```
cd ~/develop
git clone https://github.com/SNLComputation/omega_h.git
cd omega_h
git checkout v9.32.1
```

environment script `~/develop/omega_h/envRhel7v0132_noCuda.sh`

```
module use /opt/scorec/spack/v0132/lmod/linux-rhel7-x86_64/Core
module load gcc mpich cmake
export MPICH_CXX=g++
```

build

```
source ~/develop/omega_h/envRhel7v0132_noCuda.sh
mkdir ~/develop/build-omegahCudaOff-rhel7
cd !$
cmake ~/develop/omega_h \
-DCMAKE_INSTALL_PREFIX=$PWD/install \
-DBUILD_SHARED_LIBS=OFF \
-DOmega_h_USE_MPI=on \
-DCMAKE_CXX_COMPILER=`which mpicxx`

make install -j16 
```

## omegahPetsc

The `parallel_box_test_nocuda` branch is required to run without cuda:

https://github.com/cwsmith/omegahPetsc/commit/1d339dfd3ffb8b3c90fc699a1e91883a6bf66696

Run the following command to switch branches:

```
cd 
git clone https://github.com/cwsmith/omegahPetsc.git
git checkout parallel_box_test_nocuda
```

environment script `~/develop/omegahPetsc/envRhel7v0132_noCuda.sh`

```
module use /opt/scorec/spack/v0132/lmod/linux-rhel7-x86_64/Core
module load gcc mpich cmake
export MPICH_CXX=g++

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:~/develop/build-omegahCudaOff-rhel7/install/lib/cmake
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:~/develop/petsc/arch-rhel7-nocuda/lib/pkgconfig
```

build

```
source ~/develop/omegahPetsc/envRhel7v0132_noCuda.sh
mkdir ~/develop/build-omegahPetsc-cudaOff
cd !$
cmake ../omegahPetsc/ -DCMAKE_CXX_COMPILER=mpicxx
make
```
