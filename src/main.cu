#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "Tensor.cuh"
#include "setup.cuh"
#include "Fields.cuh"
#include "Preproc.cuh"



int main(){

    using namespace setup;

    //just some silly test of data by now

    SampleIdx it = 2;
    SampleIdx iz = 4;

    FullFields<MemoryOn::Host> F;
    FullProfiles<MemoryOn::Host> R;

    F.read("../data/", it, true);
    R.readAndPreproc("../data/");
    
    SlicedFields<MemoryOn::Device> Fs;
    SlicedProfiles<MemoryOn::Device> Rs;
    Preproc<MemoryOn::Device> P;

    F.copySliceWithHalo(Fs, iz);
    R.copySlice(Rs, iz);
    P.compute(Fs, Rs);

    checkResiduals(Fs, Rs, P);

    std::printf("done\n");

    return 0;
}


