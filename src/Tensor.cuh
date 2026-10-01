#pragma once
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include<array>
#include <netcdf>
#include <vector>
#include <stdexcept>
#include "MemoryOwner.cuh"

#define FORCE_INLINE inline __attribute__((always_inline))
#define HD __host__ __device__

using Index = int;

struct Coord {
    std::string name;
    int start = 0;
    int end = -1;       // inclusive; -1 = last element
    int step = 1;
};


template<typename T, std::size_t... N>
class TensorView {

protected:
    T* __restrict__ data;
    //public access via ptr()

public:

    static constexpr std::size_t rank {sizeof...(N)};
    static constexpr std::size_t size {(N *...)};
    static constexpr std::array  shape{ N...};

    static constexpr std::size_t bytesData = size * sizeof(T);
    using Indices = std::array<Index, rank>;

    using ElementType = T;
   
    HD static constexpr Indices computeColumnMajorStride(){
        Indices stride{};
        for(int dim = 0; dim < rank; dim++) {
            stride[dim] = 1;
            for(int i = 0; i < dim; i++)
                stride[dim] *= shape[i];
        }
        return stride;
    };

    static constexpr Indices memoryStride = computeColumnMajorStride();

    // =============================== Constructors ====================================== //    
    HD constexpr TensorView(T* data) : data{data} {};    
    
    
    template<std::size_t... N2>
    HD constexpr explicit TensorView(TensorView<T, N2...> t) : data{t.ptr()} {
        static_assert(t.size >= size);
    };

    // =============================== Accessors ========================================= //

    //raw pointer to a given element
    template<typename... IdxPack>
    FORCE_INLINE HD T* __restrict__ ptr(IdxPack... Is) {
        return &data[toLinIdx(Is...)];
    }
    
    template<typename... IdxPack>
    FORCE_INLINE HD const T* __restrict__ ptr(IdxPack... Is) const {
        return &data[toLinIdx(Is...)];
    }

    //raw pointer to first element
    FORCE_INLINE HD T* __restrict__ ptr() {
        return data;
    }
    
    //raw pointer to first element
    FORCE_INLINE HD const T* __restrict__ ptr() const {
        return data;
    }

    HD FORCE_INLINE auto operator[](Index iOuterDim) {
        return strippedView(std::make_index_sequence<rank-1>{}, iOuterDim);
    }

    HD FORCE_INLINE const auto operator[](Index iOuterDim) const {
        return strippedView(std::make_index_sequence<rank-1>{}, iOuterDim);
    }


    //multidimensional accessor
    template<typename... IdxPack>
    FORCE_INLINE HD constexpr 
    std::enable_if_t<(std::is_integral_v<IdxPack> && ...), //SFINAE guard 
    T&> operator()(IdxPack... Is) {
        static_assert(sizeof...(Is) == rank, "incorrect number of indices");
        return *ptr(Is...);
        //return data[toLinIdx(Is...)];
    }

    template<typename... IdxPack>
    FORCE_INLINE HD constexpr 
    std::enable_if_t<(std::is_integral_v<IdxPack> && ...), 
    const T&> operator()(IdxPack... Is) const {
        static_assert(sizeof...(Is) == rank, "incorrect number of indices");
        return *ptr(Is...);
        //return data[toLinIdx(Is...)];
    }

    // =============================== Low level methods ================================= //


    HD static constexpr Index getStride(std::size_t dim) {
        const std::size_t shape_arr[] = { N... };
        Index stride = 1;

        //ColumnMajor
        for (std::size_t i = 0; i < dim; ++i) {
            stride *= shape_arr[i];
        }
        return stride;
    }

    template<typename... IdxPack>
    FORCE_INLINE HD static constexpr Index toLinIdx(IdxPack... Is) {
        static_assert(sizeof...(Is) == rank, "Incorrect number of indices");
        static_assert((std::is_integral_v<IdxPack> && ...), "Non-integral indices");

        const Index indices[] = { static_cast<Index>(Is)... };
        Index linIdx = 0;

        for (std::size_t dim = 0; dim < rank; dim++) {
            linIdx += indices[dim] * getStride(dim);
        }

        return linIdx;
    }

    private:
    template<std::size_t I>
    static constexpr std::size_t dim() noexcept {
        return shape[I];
    }

    template<std::size_t... Is>
    HD FORCE_INLINE auto strippedView(std::index_sequence<Is...>, Index iOuterDim) {
        return TensorView<T, dim<Is>()...>(ptr((Is * 0)..., iOuterDim));
    }


    template<std::size_t... Is>
    HD FORCE_INLINE const auto strippedView(std::index_sequence<Is...>, Index iOuterDim) const {
        //ugly cast but lets leave it by now
        return TensorView<T, dim<Is>()...>(const_cast<T*>(ptr((Is * 0)..., iOuterDim)));
    }

    public:

    void fromNetCDF(const std::string& fileName, const std::string& varName, std::vector<Coord> coords, bool verbose = false)
    {
        if(verbose) {
            std::printf("from %s reading %s", fileName.c_str(), varName.c_str());
            std::fflush(stdout);
        }
        netCDF::NcFile f(fileName, netCDF::NcFile::read);

        netCDF::NcVar var = f.getVar(varName);
        if(var.isNull())                          throw std::runtime_error(std::string(varName) + " not found\n");
        if(var.getDims().size() != coords.size()) throw std::runtime_error("not enough dimensions specified\n");
        std::vector<netCDF::NcDim> dims = var.getDims();

        std::vector<std::size_t>    start(coords.size());
        std::vector<std::size_t>    count(coords.size());
        std::vector<std::ptrdiff_t> step (coords.size());

        int tensorDim = rank - 1; //rev. order cause column (Tensor) vs row (Nc) major order
        for(int i = 0; i < dims.size(); i++){
            std::string istr = "dim: " + std::to_string(i) + " ";
            if(dims[i].getName() != coords[i].name)
                throw std::runtime_error(istr + "name mismatch: " + dims[i].getName() + " != " + coords[i].name);
            
            if(coords[i].end == -1) coords[i].end = dims[i].getSize()-1;
            start[i] =  coords[i].start;
            count[i] = (coords[i].end - coords[i].start) / coords[i].step + 1;
            step [i] =  coords[i].step;

            //sanity checks
            if( coords[i].end  >= dims[i].getSize() || coords[i].end < 0) throw std::runtime_error(istr + "invalid end index\n");        
            if( coords[i].start > coords[i].end || coords[i].start < 0)   throw std::runtime_error(istr + "invalid start index\n"); 
            if( coords[i].step < 1)                                       throw std::runtime_error(istr + "step must be positive\n");
            if((coords[i].end - coords[i].start) % coords[i].step != 0)   throw std::runtime_error(istr + "end is not reachable with step\n");

            if(verbose){
                std::printf("[%s = %d:%d:%d]", coords[i].name.c_str(), coords[i].start, coords[i].end, coords[i].step);
                std::fflush(stdout);
            }

            if(tensorDim >= 0 && count[i] == shape[tensorDim])
                tensorDim--;
            else if(count[i] != 1) //slice is fine
                throw std::runtime_error("dimension mismatch: " + std::to_string(count[i]) + " != " + std::to_string(shape[tensorDim])); 
        }

        var.getVar(start, count, step, ptr());
        if(verbose) std::printf(" done.\n");
    }

    void toNetCDF(const std::string& fileName, const std::string& varName,
                std::vector<Coord> coords, bool append = true, bool verbose = false){
        if(verbose) {
            std::printf("to %s writing %s", fileName.c_str(), varName.c_str());
            std::fflush(stdout);
        }

        netCDF::NcFile f(fileName, append ? netCDF::NcFile::write: netCDF::NcFile::replace);

        std::vector<std::size_t>    start(coords.size());
        std::vector<std::size_t>    count(coords.size());
        std::vector<std::ptrdiff_t> step (coords.size());
        std::vector<netCDF::NcDim>  dims (coords.size());

        for(int i = 0; i < coords.size(); i++){
            if(verbose){
                std::printf("[%s = %d:%d:%d]", coords[i].name.c_str(), coords[i].start, coords[i].end, coords[i].step);
                std::fflush(stdout);
            }

            std::string istr = "dim: " + std::to_string(i) + " ";

            int tensorDim = rank - i - 1;
            std::size_t dimSize = shape[tensorDim];

            dims[i] = f.getDim(coords[i].name);

            if(dims[i].isNull())
                dims[i] = f.addDim(coords[i].name, dimSize);
            else if(dims[i].getSize() != dimSize)
                throw std::runtime_error(istr + "dimension size mismatch");

            if(coords[i].end == -1) coords[i].end = dimSize - 1;

            //sanity checks
            if( coords[i].end >= dimSize || coords[i].end < 0)          throw std::runtime_error(istr + "invalid end index\n");
            if( coords[i].start > coords[i].end || coords[i].start < 0) throw std::runtime_error(istr + "invalid start index\n");
            if( coords[i].step < 1)                                     throw std::runtime_error(istr + "step must be positive\n");
            if((coords[i].end - coords[i].start) % coords[i].step != 0) throw std::runtime_error(istr + "end is not reachable with step\n");

            start[i] =  coords[i].start;
            count[i] = (coords[i].end - coords[i].start) / coords[i].step + 1;
            step [i] =  coords[i].step;
        }

        netCDF::NcVar var = f.getVar(varName);

        if(!var.isNull()) throw std::runtime_error(std::string(varName) + " already exists\n");

        var = f.addVar(varName, netCDF::ncDouble, dims);

        var.putVar(start, count, step, ptr());
        if(verbose) std::printf(" done.\n");
    }

};

template<MemoryOn Location, typename T, std::size_t... N>
using Tensor = Owning<Location, TensorView<T, N...>>;


