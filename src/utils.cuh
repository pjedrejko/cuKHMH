#pragma once

using IdxRange = std::array<Index, 2>;
    
template<std::size_t Nxt, std::size_t Nyt, std::size_t Nzt, std::size_t Nxs, std::size_t Nys, std::size_t Nzs>
void copyField(
          TensorView<double, Nxt, Nyt, Nzt> trg,
    const TensorView<double, Nxs, Nys, Nzs> src, 
    IdxRange is = {0, -1}, IdxRange js = {0, -1}, IdxRange ks = {0, -1}, 
    std::array<int, 3> halo = {0, 0, 0}){

    if(is[1] == -1) is[1] = Nxs;
    if(js[1] == -1) js[1] = Nys;
    if(ks[1] == -1) ks[1] = Nzs;
    
    assert((is[1] - is[0] + 2 * halo[0]) == Nxt && "DIM X MISMATCH");
    assert((js[1] - js[0] + 2 * halo[1]) == Nyt && "DIM Y MISMATCH");
    assert((ks[1] - ks[0] + 2 * halo[2]) == Nzt && "DIM Z MISMATCH");

    std::size_t s = sizeof(double);
    cudaPitchedPtr srcPtr = make_cudaPitchedPtr(const_cast<double*>(src.ptr()), Nxs*s, Nxs*s, Nys);
    cudaPitchedPtr dstPtr = make_cudaPitchedPtr(const_cast<double*>(trg.ptr()), Nxt*s, Nxt*s, Nyt);

    cudaPos     srcPos = make_cudaPos(  is[0]*s,   js[0],   ks[0]);
    cudaPos     dstPos = make_cudaPos(halo[0]*s, halo[1], halo[2]);

    cudaExtent  extent = make_cudaExtent( (is[1] - is[0])*s, js[1] - js[0], ks[1] - ks[0]);

    cudaMemcpy3DParms p = {0};
    p.srcPtr   = srcPtr;
    p.srcPos   = srcPos;
    p.dstPtr   = dstPtr;
    p.dstPos   = dstPos;
    p.extent   = extent;
    p.kind     = cudaMemcpyDefault;

    cudaError_t err = cudaMemcpy3DAsync(&p);
    if (err != cudaSuccess)
        std::fprintf(stderr, "copyField:: failed: %s\n", cudaGetErrorString(err));
}


template<std::size_t Nxt, std::size_t Nyt, std::size_t Nxs, std::size_t Nys>
void copyField(
          TensorView<double, Nxt, Nyt> trg,
    const TensorView<double, Nxs, Nys> src, 
    IdxRange is = {0, -1}, IdxRange js = {0, -1}, 
    std::array<int, 2> halo = {0, 0}){
    copyField(TensorView<double, Nxt, Nyt, 1>(trg), TensorView<double, Nxs, Nys, 1>(src), is, js, {0, 1}, {halo[0], halo[1], 0});
    }

template<std::size_t Nxt, std::size_t Nxs>
void copyField(
          TensorView<double, Nxt> trg,
    const TensorView<double, Nxs> src, 
    IdxRange is = {0, -1},
    std::array<int, 1> halo = {0}){
    copyField(TensorView<double, Nxt, 1>(trg), TensorView<double, Nxs, 1>(src), is, {0, 1}, {halo[0], 0});
    }


//by now put it here
template<typename T, std::size_t... N>
void copyTensor(TensorView<T, N...> trg, TensorView<T, N...> src){
      cudaError_t err = cudaMemcpyAsync(trg.ptr(), src.ptr(), src.bytesData, cudaMemcpyDefault, 0);

    if (err != cudaSuccess)
        std::fprintf(stderr, "copyField:: failed: %s\n", cudaGetErrorString(err));
}