(cd $(dirname $0)/.. &&
    python3 -m tarfile -c gpgpu-sim-modified.tar.gz . && 
    mkdir -p spack-mirror/gpgpu-sim &&
    mv gpgpu-sim-modified.tar.gz spack-mirror/gpgpu-sim/ )

spack mirror add gpgpu-sim file://$(realpath $(dirname $0)/../spack-mirror)
spack uninstall -y gpgpu-sim@modified | cat # 这样不存在也不会中断
spack install -y --no-checksum gpgpu-sim@modified%gcc@7.5.0 ^ mesa+glx~llvm
spack mirror rm gpgpu-sim
rm -rf spack-mirror
