#pragma once
#include <cstddef>


template<int R, int C, typename T>
struct Kernel2D {
    static constexpr int rows = R;
    static constexpr int cols = C;
    static constexpr int ay = R / 2;
    static constexpr int ax = C / 2;

    const T data[R][C];

    constexpr T operator()(int y, int x) const {
        return data[y][x];
    }
};

template<typename Kernel, typename T>
inline void convolve_ignore_border(
    const T* src, T* dst,
    int w, int h, int stride,
    const Kernel& k
) {
    for (int y = Kernel::ay; y < h - Kernel::ay; ++y) {
        for (int x = Kernel::ax; x < w - Kernel::ax; ++x) {
            T acc = T(0);

            #pragma unroll
            for (int ky = 0; ky < Kernel::rows; ++ky)
                #pragma unroll
                for (int kx = 0; kx < Kernel::cols; ++kx)
                    acc += src[(y + ky - Kernel::ay) * stride +
                               (x + kx - Kernel::ax)] * k(ky, kx);

            dst[y * stride + x] = acc;
        }
    }
}