#pragma once
#include "Tensor.cuh"
#include <array>

enum Stag: int{UGrid = 0, VGrid = 1, WGrid = 2, PGrid = 3};

//for face indices which are int+-0.5. e.g. face between cell 1 and 2 has index = 1.5, so IndexDoubled = 3
using IndexDoubled = Index;

namespace autoFVM{
    namespace details{
        //takes IndexDoubled (of e.g. face) but returns normal Index (of a cell)!
        HD FORCE_INLINE constexpr Index closestPrevCell(IndexDoubled i) {
            // iPrev = floor(i / 2.0)
            // For -7 (representing -3.5), -7 >> 1 results in -4
            return i >> 1;
        }

        //takes IndexDoubled (of e.g. face) but returns normal Index (of a cell)!
        HD FORCE_INLINE constexpr Index closestNextCell(IndexDoubled i) {
            // iNext = ceil(i / 2.0)
            // For -7 (representing -3.5), (-7+1) >> 1 results in -3
            return (i + 1) >> 1;
        }

        struct Neighbours {
            std::array<std::array<Index, 3>, 8> idx;
            std::array<int, 8> weight{};
            int count = 0;
        };

        //generates a unique set of cell indices from index span. Repetitions included in weight
        HD constexpr details::Neighbours generateNeighbours(std::array<std::array<Index, 2>, 3> idxSpan){
            using namespace details;
            Neighbours unique{};

            //take all the configurations
            for (int ix : idxSpan[0]) {
                for (int iy : idxSpan[1]) {
                    for (int iz : idxSpan[2]) {
                        int foundIdx = -1;
                        //linear search for uniqueness
                        for (int i = 0; i < unique.count; ++i) {
                            if (unique.idx[i][0] == ix && unique.idx[i][1] == iy && unique.idx[i][2] == iz ) {
                                foundIdx = i;
                                break;
                            }
                        }
                        
                        //if unique add it, if not, increase weight
                        if (foundIdx != -1) {
                            unique.weight[foundIdx]++;
                        } else {
                            unique.idx[unique.count] = {ix, iy, iz};
                            unique.weight[unique.count] = 1;
                            unique.count++;
                        }
                    }
                }
            }
            return unique;
    }

        //returns stagger of the sampling point with respect to the grid of the field sampled
        HD constexpr std::array<IndexDoubled, 3> sampleRelativeStagger(Stag StagTrg, 
            std::array<IndexDoubled, 3> samplePointRelCoords, Stag StagSrc){
            using namespace details;

            std::array<IndexDoubled, 3> h = {0, 0, 0}; 
            //whole indices assumed on P grid
            
            //if we want to sample on U-grid, its nodes are h/2 to the left
            if(StagTrg != Stag::PGrid)
                h[StagTrg]--;

            //if we want to sample on the left face of that U-grid, it is another h/2 to the left
            h[0] += samplePointRelCoords[0];
            h[1] += samplePointRelCoords[1];
            h[2] += samplePointRelCoords[2];

            //if we sample sth elese (e.g. V) move the coordinates to make V nodes in the whole indices
            if(StagSrc != Stag::PGrid)
                h[StagSrc]++;

            return h;
        }
        
    }//details
    
    template<Stag StagTrg, IndexDoubled ShiftX, IndexDoubled ShiftY, IndexDoubled ShiftZ, Stag Src>
    struct InterpSetup{};

    template<Stag StagTrg, Stag StagSrc>
    using InCenter = InterpSetup<StagTrg, 0, 0, 0, StagSrc>;

    template<Stag StagTrg, std::size_t FaceDir, std::size_t Side, Stag StagSrc>
    using OnFace = InterpSetup<StagTrg, 
    (FaceDir == 0)? (IndexDoubled(Side) * 2 - 1): 0, 
    (FaceDir == 1)? (IndexDoubled(Side) * 2 - 1): 0, 
    (FaceDir == 2)? (IndexDoubled(Side) * 2 - 1): 0,
    StagSrc>;

    template<Stag StagSrc, Stag StagTrg, IndexDoubled ShiftX, IndexDoubled ShiftY, IndexDoubled ShiftZ,
        std::size_t Nx, std::size_t Ny, std::size_t Nz>
    HD FORCE_INLINE double interp(
        InterpSetup<StagTrg, ShiftX, ShiftY, ShiftZ, StagSrc>,
        const TensorView<double, Nx, Ny, Nz> F, 
        Index ix, Index iy, Index iz){
        using namespace details;
        constexpr std::array<IndexDoubled, 3> h = sampleRelativeStagger(StagTrg, 
            {ShiftX, ShiftY, ShiftZ}, StagSrc);

        constexpr std::array<std::array<Index, 2>, 3> idxSpan = {{
        {{closestPrevCell(h[0]), closestNextCell(h[0])}},
        {{closestPrevCell(h[1]), closestNextCell(h[1])}},
        {{closestPrevCell(h[2]), closestNextCell(h[2])}}
        }};
        constexpr Neighbours nbr = generateNeighbours(idxSpan); //relative indices
        constexpr double totalWeight = 8;

        double res = 0.0;
        #pragma unroll
        for(int i = 0; i < nbr.count; i++)
            res += (F)(ix + nbr.idx[i][0], iy + nbr.idx[i][1], iz + nbr.idx[i][2]) * nbr.weight[i];

        return res / totalWeight;
    }

    template<std::size_t DiffDir, Stag StagSrc, Stag StagTrg, IndexDoubled ShiftX, IndexDoubled ShiftY, IndexDoubled ShiftZ,
        std::size_t Nx, std::size_t Ny, std::size_t Nz>
    HD FORCE_INLINE double diff(
        InterpSetup<StagTrg, ShiftX, ShiftY, ShiftZ, StagSrc>,
        const TensorView<double, Nx, Ny, Nz> F, 
        Index ix, Index iy, Index iz){
        using namespace details;
 
        constexpr std::array<IndexDoubled, 3> dx{DiffDir==0, DiffDir==1, DiffDir==2};

        return
        ( interp(InterpSetup<StagTrg, ShiftX+dx[0], ShiftY+dx[1], ShiftZ+dx[2], StagSrc>{}, F, ix, iy, iz)
         -interp(InterpSetup<StagTrg, ShiftX-dx[0], ShiftY-dx[1], ShiftZ-dx[2], StagSrc>{}, F, ix, iy, iz) );

    }
}