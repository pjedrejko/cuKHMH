#pragma once
#include<array>
#include<string>
namespace setup{

    //domain size
    static constexpr std::size_t NX = 51;
    static constexpr std::size_t NY = 51;
    static constexpr std::size_t NZ = 32;

    //ghost nodes
    static constexpr std::size_t GX = 2;
    static constexpr std::size_t GY = 2;
    static constexpr std::size_t GZ = 2;

    static constexpr std::array<double, 3> dxInv{0.01, 0.01, 0.01};
    static constexpr std::array<double, 3> utrans{-8.0, 0.0, 0.0};
    static constexpr std::array<double, 2> nj{-1.0, 1.0};

    static constexpr double nu = 1e-5;
    static constexpr double fCorio = 3.76e-5;

}