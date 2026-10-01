#pragma once
#include<array>


#define FORCE_INLINE inline __attribute__((always_inline))
#define HD __host__ __device__


template<std::size_t... Is, typename LambdaT>
HD FORCE_INLINE constexpr void for_iteration(std::index_sequence<Is...>, const LambdaT& body) {
    (body.template operator()<Is>(), ...);
}

template<std::size_t N, typename LambdaT>
HD FORCE_INLINE constexpr void for_constexpr(const LambdaT& body) {
    for_iteration(std::make_index_sequence<N>{}, body);
}

//  EXAMPLE:
// for_constexpr<3>([&]<std::size_t i>() {
//        std::get<i>(...)
// });
