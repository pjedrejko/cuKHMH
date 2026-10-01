#pragma once
#include "setup.cuh"
#include "Tensor.cuh"
#include "Fields.cuh"
#include "autoFVM.cuh"
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

using OnFacesFieldView  = TensorView<double, setup::NX, setup::NY, nSides, nFaces, nGrids>;
using InCenterFieldView = TensorView<double, setup::NX, setup::NY, nGrids>; 
using OnFacesProfView   = TensorView<double, nSides, nFaces, nGrids>;

struct PreprocView;

__global__ void preprocAdvc(const SlicedFieldsView f, const SlicedProfilesView r, PreprocView p);
__global__ void preprocVisc(const SlicedFieldsView f, const SlicedProfilesView r, PreprocView p);
__global__ void preprocPres(const SlicedFieldsView f, const SlicedProfilesView r, PreprocView p);


struct PreprocView{

    OnFacesFieldView  Uj, Ui, Sij, Nut;
    InCenterFieldView Advc, Visc, Pres;

    PreprocView(double* memPtr):
        Uj  (memPtr),
        Ui  (  Uj.ptr() +   Uj.size),
        Sij (  Ui.ptr() +   Ui.size),
        Nut ( Sij.ptr() +  Sij.size),
        Advc( Nut.ptr() +  Nut.size),
        Visc(Advc.ptr() + Advc.size),
        Pres(Visc.ptr() + Visc.size)
    {}

    static constexpr std::size_t bytesData =  4 * OnFacesFieldView::bytesData + 3 * InCenterFieldView::bytesData;

    using ElementType = double;

    void compute(SlicedFieldsView f, SlicedProfilesView r){
        using namespace setup;
        dim3 block(16, 16);
        dim3 grid((NX+block.x-1)/block.x, (NY+block.y-1)/block.y);

        preprocAdvc<<<grid, block>>>(f, r, *this);
        preprocVisc<<<grid, block>>>(f, r, *this);
        preprocPres<<<grid, block>>>(f, r, *this);
    }

};

template<MemoryOn Location>
using Preproc = Owning<Location, PreprocView>;


__global__
void preprocAdvc(
    const SlicedFieldsView f,
    const SlicedProfilesView r,
    PreprocView p
){
    using namespace autoFVM;
    using namespace setup;
    
    SampleIdx ix = blockIdx.x * blockDim.x + threadIdx.x;
    SampleIdx iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= NX || iy >= NY) return;
        for_constexpr<nGrids>([&]<GridIdx i>(){
            p.Advc(ix, iy, i) = 0.0;
        for_constexpr<nFaces>([&]<FaceIdx j>(){
        for_constexpr<nSides>([&]<SideIdx s>(){
            OnFace< Stag(i), j,s,  Stag(i)> i2i;
            OnFace< Stag(i), j,s,  Stag(j)> j2i;
            p.Ui   (ix,iy, s, j, i) = interp (i2i, f.U[i], ix+1, iy+1, 0+1); //+1 cause ghosts
            p.Uj   (ix,iy, s, j, i) = interp (j2i, f.U[j], ix+1, iy+1, 0+1);
            p.Advc (ix,iy, i)      += p.Ui(ix, iy, s, j, i) * p.Uj(ix, iy, s, j, i) * r.RhoCoef(0, s, j, i) * nj[s] * dxInv[j];
        });
        });
        });
}


__global__
void preprocVisc(
    const SlicedFieldsView f,
    const SlicedProfilesView r,
    PreprocView p
){
    using namespace autoFVM;
    using namespace setup;
    
    SampleIdx ix = blockIdx.x * blockDim.x + threadIdx.x;
    SampleIdx iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= NX || iy >= NY) return;
        for_constexpr<3>([&]<GridIdx i>(){
            p.Visc(ix, iy, i) = 0.0;
        for_constexpr<3>([&]<FaceIdx j>(){
        for_constexpr<2>([&]<SideIdx s>(){
            OnFace< Stag(i), j,s,  Stag(i)>    i2i;
            OnFace< Stag(i), j,s,  Stag(j)>    j2i;
            OnFace< Stag(i), j,s, Stag::PGrid> p2i;
            p.Nut(ix,iy,  s,  j,i) =    interp(p2i, f.Nut,  ix+1, iy+1, 0+1);
            p.Sij(ix,iy,  s,  j,i) = ( diff<i>(j2i, f.U[j], ix+1, iy+1, 0+1)*dxInv[i] 
                                      +diff<j>(i2i, f.U[i], ix+1, iy+1, 0+1)*dxInv[j] ) * 0.5;

            p.Visc  (ix,iy, i)      += 2 * p.Sij(ix, iy, s, j, i) * (p.Nut(ix, iy, s, j, i) + nu) * r.RhoCoef(0, s, j, i) * nj[s] * dxInv[j];
        });
        });
        });
}


HD FORCE_INLINE constexpr double eijk(int i, int j, int k) {
    //Levi-Civita symbol
    if ( i == j || j == k || i == k) return 0.0;
    if ((i == 0 && j == 1 && k == 2) || 
        (i == 1 && j == 2 && k == 0) || 
        (i == 2 && j == 0 && k == 1)) return 1.0;
    return -1.0;
}

__global__
void preprocPres(
    const SlicedFieldsView f,
    const SlicedProfilesView r,
          PreprocView p
){
    using namespace autoFVM;
    using namespace setup;

    SampleIdx ix = blockIdx.x * blockDim.x + threadIdx.x;
    SampleIdx iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;
        for_constexpr<3>([&]<GridIdx i>(){
            InCenter< Stag(i), Stag::PGrid> p2i;
            p.Pres(ix,iy, i) = -diff<i>(p2i, f.P, ix+1, iy+1, 0+1)*dxInv[i];
        });
        for_constexpr<2>([&]<GridIdx i>(){
            constexpr GridIdx j = (i+1) % 2;
            InCenter< Stag(i), Stag(j)> j2i;
            p.Pres(ix,iy, i) -= eijk(i,2,j) * fCorio * (interp(j2i, f.U[j], ix+1, iy+1, 0+1) + utrans[j] - r.Ug[j](0));
        });
}



__global__
void navierStokesResidual(
    const SlicedFieldsView f,
    const PreprocView p,
    TensorView<double, setup::NX, setup::NY, nGrids> out
){
    using namespace setup;

    SampleIdx ix = blockIdx.x * blockDim.x + threadIdx.x;
    SampleIdx iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;
        for_constexpr<3>([&]<GridIdx i>(){

            out(ix, iy, i) = std::abs( 
            - f.Ut(ix+1, iy+1, 0+1, i)
            - p.Advc(ix, iy, i) 
            + p.Visc(ix, iy, i) 
            + p.Pres(ix, iy, i)
            + ((i==2)? f.B(ix+1, iy+1, 0+1): 0)
            );
        });
}


__global__
void continuityResidual(
    const SlicedFieldsView f,
    const SlicedProfilesView r,
    TensorView<double, setup::NX, setup::NY> out
){
    using namespace setup;
    using namespace autoFVM;

    SampleIdx ix = blockIdx.x * blockDim.x + threadIdx.x;
    SampleIdx iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;
        out(ix, iy) = 0;
        for_constexpr<nGrids>([&]<GridIdx i>(){
        for_constexpr<nSides>([&]<SideIdx s>(){
            OnFace<PGrid, i, s, Stag(i)> i2p;
            out(ix, iy) += interp (i2p, f.U[i], ix+1, iy+1, 0+1) * r.RhoCoef(0, s, i, 0) * nj[s] * dxInv[i]; //+1 cause ghosts
        });
        });
        out(ix, iy) = std::abs(out(ix, iy));
}

void checkResiduals(
    const SlicedFieldsView fs,
    const SlicedProfilesView rs,
    const PreprocView p
){
    using namespace setup;
    Tensor<MemoryOn::Device, double, NX, NY, 3> resNS; 
    Tensor<MemoryOn::Device, double, NX, NY>    resCont; 

    dim3 block(16, 16);
    dim3 grid((NX+block.x-1)/block.x, (NY+block.y-1)/block.y);


    navierStokesResidual<<<block, grid>>>(fs, p, resNS);
    continuityResidual  <<<block, grid>>>(fs, rs, resCont);
   
    std::printf("max local residual:\n");

    for(GridIdx i = 0; i < nGrids; i++)
        std::printf("\tNavier-Stokes[%d]: % .16e\n", i,
            thrust::reduce(thrust::device, resNS[i].ptr(), resNS[i].ptr()+resNS[i].size, -1.0, thrust::maximum<double>{}));

    std::printf("\tcontinuity:       % .16e\n",
        thrust::reduce(thrust::device, resCont.ptr(), resCont.ptr()+resCont.size, -1.0, thrust::maximum<double>{}));

}

