#pragma once
#include "Tensor.cuh"
#include "for_constexpr.cuh"
#include "setup.cuh"
#include "utils.cuh"

using GridIdx = int;
using FaceIdx = int;
using SideIdx = int;
using SampleIdx = int;
using FieldIdx = int;

static constexpr FaceIdx nFaces = 3;
static constexpr GridIdx nGrids = 3;
static constexpr SideIdx nSides = 2;

template<std::size_t Nx, std::size_t Ny, std::size_t Nz>
struct FieldsView{
    TensorView<double, Nx, Ny, Nz, nGrids> U, Ut;
    TensorView<double, Nx, Ny, Nz> P, B, Nut;
    static constexpr std::size_t nFields = 9;

    static constexpr std::size_t bytesData = nFields * Nx * Ny * Nz * sizeof(double);
    using ElementType = double;
    using StackedType = TensorView<double, Nx, Ny, Nz, nFields>;
    StackedType stacked;

    FieldsView(double* memPtr):
        U  (memPtr),
        Ut ( U.ptr() +  U.size),
        P  (Ut.ptr() + Ut.size),
        B  ( P.ptr() +  P.size),
        Nut( B.ptr() +  B.size),
        stacked(memPtr)
        {}
};

template<MemoryOn Location, std::size_t Nx, std::size_t Ny, std::size_t Nz>
using Fields = Owning<Location, FieldsView<Nx, Ny, Nz>>;

template<std::size_t Nz>
struct ProfilesView{
    TensorView<double, Nz, nSides, nFaces, nGrids> RhoCoef;
    TensorView<double, Nz, nGrids> Ug;

    static constexpr std::size_t nFields = 21;
    static constexpr std::size_t bytesData = nFields * Nz * sizeof(double);
    using StackedType = TensorView<double, Nz, nFields>;
    using ElementType = double;
    StackedType stacked;

    ProfilesView(double* memPtr):
        RhoCoef(memPtr),
        Ug(RhoCoef.ptr() + RhoCoef.size),
        stacked(memPtr)
        {}
};

using SlicedProfilesView = ProfilesView<1>;

template<MemoryOn Location>
using SlicedProfiles = Owning<Location, SlicedProfilesView>;

struct FullProfilesView: ProfilesView<setup::NZ>{

    FullProfilesView(double* memPtr): ProfilesView<setup::NZ>(memPtr){};

    void copySlice(SlicedProfilesView sp, SampleIdx iz){
        //not very efficient, but called rarely and keeps the interface uniform
        copyField(sp.stacked, stacked, {iz, iz+1}, {0, -1}); 
    }

    void readAndPreproc(const std::string& dataDir){
        using namespace setup;
        //temporary buffers
        Tensor<MemoryOn::Host, double, NZ  > rho;
        Tensor<MemoryOn::Host, double, NZ+1> rhoh;

        rho. fromNetCDF(dataDir + "thermo_basestate.nc", "rhoref",  {Coord{"k"}}       , true);
        rhoh.fromNetCDF(dataDir + "thermo_basestate.nc", "rhorefh", {Coord{"k_plus1"}},  true);

        Ug[0].fromNetCDF(dataDir + "bomex_input.nc", "u_geo", {Coord{"z"}}, true);
        Ug[1].fromNetCDF(dataDir + "bomex_input.nc", "v_geo", {Coord{"z"}}, true);

        //sample rho on faces
        for(SampleIdx iz = 1; iz < (NZ-1); iz++){//neglect BC by now    
            for(GridIdx i = 0; i < 2; i++){
                for(FaceIdx j = 0; j < 2; j++)
                    for(SideIdx s = 0; s < 2; s++)
                        RhoCoef(iz, s, j, i) = rho(iz);

                RhoCoef(iz, 0, 2, i) = rhoh(iz);
                RhoCoef(iz, 1, 2, i) = rhoh(iz+1);
            }

            for(FaceIdx j = 0; j < 2; j++)
                for(SideIdx s = 0; s < 2; s++)
                    RhoCoef(iz, s, j, 2) = rhoh(iz);

            RhoCoef(iz, 0, 2, 2) = rho(iz-1);
            RhoCoef(iz, 1, 2, 2) = rho(iz);
        }

        //normalize by rho in the center
        for(SampleIdx iz = 1; iz < (NZ-1); iz++){
            for(GridIdx i = 0; i < nGrids; i++){
                double rhoCenter = RhoCoef(iz, 0, 0, i);
                for(FaceIdx j = 0; j < nFaces; j++){
                    for(SideIdx s = 0; s < nSides; s++)
                        RhoCoef(iz, s, j, i) /= rhoCenter;
                }
            }
        }
    }
};

template<MemoryOn Location>
using FullProfiles = Owning<Location, FullProfilesView>;

using SlicedFieldsView = FieldsView<setup::NX+2, setup::NY+2, 1+2>;

template<MemoryOn Location>
using SlicedFields = Owning<Location, SlicedFieldsView>;

__global__
void periodicHaloXY(SlicedFieldsView FS){
    using namespace setup;
    SampleIdx it = blockIdx.x * blockDim.x + threadIdx.x;

    if (it < setup::NY+2){ //halo along y
        for_constexpr<FS.nFields>([&]<FieldIdx f>(){
        for_constexpr<1+2>       ([&]<SampleIdx iz>(){
            FS.stacked(0,    it, iz, f) = FS.stacked(NX, it, iz, f); 
            FS.stacked(NX+1, it, iz, f) = FS.stacked(1,  it, iz, f); 
        });
        });
    }

    if (it < NX+2){ //halo along x
        for_constexpr<FS.nFields>([&]<FieldIdx f>(){
        for_constexpr<1+2>       ([&]<SampleIdx iz>(){
            FS.stacked(it, 0,    iz, f) = FS.stacked(it, NY, iz, f); 
            FS.stacked(it, NY+1, iz, f) = FS.stacked(it, 1, iz,  f); 
        });
        });
    }
}


struct FullFieldsView: FieldsView<setup::NX, setup::NY, setup::NZ>{
    
    FullFieldsView(double* memPtr): FieldsView<setup::NX, setup::NY, setup::NZ>(memPtr){};

    void read(const std::string& dataDir, SampleIdx timeStep, bool verbose = false){
        Coord time{"time", timeStep, timeStep};
        std::vector<Coord> uGridCoords = {time, {"z"}, {"y"},{"xh"}};
        std::vector<Coord> vGridCoords = {time, {"z"}, {"yh"},{"x"}};
        std::vector<Coord> wGridCoords = {time, {"zh"},{"y"}, {"x"}};
        std::vector<Coord> pGridCoords = {time, {"z"}, {"y"}, {"x"}};

        U [0].fromNetCDF(dataDir + "u.nc",    "u",     uGridCoords, verbose);
        U [1].fromNetCDF(dataDir + "v.nc",    "v",     vGridCoords, verbose);
        U [2].fromNetCDF(dataDir + "w.nc",    "w",     wGridCoords, verbose);
        Ut[0].fromNetCDF(dataDir + "dtu.nc",  "dtu",   pGridCoords, verbose);
        Ut[1].fromNetCDF(dataDir + "dtv.nc",  "dtv",   pGridCoords, verbose);
        Ut[2].fromNetCDF(dataDir + "dtw.nc",  "dtw",   pGridCoords, verbose);
        B.    fromNetCDF(dataDir + "bh.nc",   "bh",    pGridCoords, verbose); //typo in nc dims names
        P.    fromNetCDF(dataDir + "p.nc",    "p",     pGridCoords, verbose);
        Nut.  fromNetCDF(dataDir + "evisc.nc","evisc", pGridCoords, verbose);
    }

    void copySliceWithHalo(SlicedFieldsView sliceWithHalo, SampleIdx iz){
        using namespace setup;

        for_constexpr<nFields>([&]<FieldIdx i>(){
            //cant be done with one copy due to halo
            copyField(sliceWithHalo.stacked[i], stacked[i], {0, -1}, {0, -1}, {iz-1, iz+2}, {1, 1, 0}); 
        });

        dim3 block(16);
        dim3 grid((std::max(NX, NY)+block.x-1)/block.x);
        periodicHaloXY<<<grid, block>>>(sliceWithHalo);
    }

};

template<MemoryOn Location>
using FullFields = Owning<Location, FullFieldsView>;



