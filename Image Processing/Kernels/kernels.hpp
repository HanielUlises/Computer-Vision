#pragma once

#include "kernel2d.hpp"


// Gradient kernels

// Sobel
constexpr Kernel2D<3,3,float> SobelX {{
    { -1,  0,  1 },
    { -2,  0,  2 },
    { -1,  0,  1 }
}};

constexpr Kernel2D<3,3,float> SobelY {{
    { -1, -2, -1 },
    {  0,  0,  0 },
    {  1,  2,  1 }
}};

// Prewitt
constexpr Kernel2D<3,3,float> PrewittX {{
    { -1,  0,  1 },
    { -1,  0,  1 },
    { -1,  0,  1 }
}};

constexpr Kernel2D<3,3,float> PrewittY {{
    { -1, -1, -1 },
    {  0,  0,  0 },
    {  1,  1,  1 }
}};

// Scharr (better rotational symmetry)
constexpr Kernel2D<3,3,float> ScharrX {{
    { -3,  0,  3 },
    { -10, 0, 10 },
    { -3,  0,  3 }
}};

constexpr Kernel2D<3,3,float> ScharrY {{
    { -3, -10, -3 },
    {  0,   0,  0 },
    {  3,  10,  3 }
}};


// Smoothing kernels

// Box blur
constexpr Kernel2D<3,3,float> Box3 {{
    { 1.f/9, 1.f/9, 1.f/9 },
    { 1.f/9, 1.f/9, 1.f/9 },
    { 1.f/9, 1.f/9, 1.f/9 }
}};

constexpr Kernel2D<5,5,float> Box5 {{
    { 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25 },
    { 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25 },
    { 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25 },
    { 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25 },
    { 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25 }
}};

// Gaussian (σ ≈ 1)
constexpr Kernel2D<3,3,float> Gaussian3 {{
    { 1.f/16, 2.f/16, 1.f/16 },
    { 2.f/16, 4.f/16, 2.f/16 },
    { 1.f/16, 2.f/16, 1.f/16 }
}};

// Gaussian (σ ≈ 1.4)
constexpr Kernel2D<5,5,float> Gaussian5 {{
    { 1,  4,  6,  4, 1 },
    { 4, 16, 24, 16, 4 },
    { 6, 24, 36, 24, 6 },
    { 4, 16, 24, 16, 4 },
    { 1,  4,  6,  4, 1 }
}};


// Second derivatives

// Laplacian (4-neighbor)
constexpr Kernel2D<3,3,float> Laplacian4 {{
    {  0,  1,  0 },
    {  1, -4,  1 },
    {  0,  1,  0 }
}};

// Laplacian (8-neighbor)
constexpr Kernel2D<3,3,float> Laplacian8 {{
    {  1,  1,  1 },
    {  1, -8,  1 },
    {  1,  1,  1 }
}};

// Second derivative X
constexpr Kernel2D<3,3,float> Dxx {{
    { 0,  0,  0 },
    { 1, -2,  1 },
    { 0,  0,  0 }
}};

// Second derivative Y
constexpr Kernel2D<3,3,float> Dyy {{
    { 0,  1,  0 },
    { 0, -2,  0 },
    { 0,  1,  0 }
}};


// Edge enhancement

// Sharpen
constexpr Kernel2D<3,3,float> Sharpen {{
    {  0, -1,  0 },
    { -1,  5, -1 },
    {  0, -1,  0 }
}};

// Strong sharpen
constexpr Kernel2D<3,3,float> SharpenStrong {{
    { -1, -1, -1 },
    { -1,  9, -1 },
    { -1, -1, -1 }
}};

// High-pass
constexpr Kernel2D<3,3,float> HighPass {{
    { -1, -1, -1 },
    { -1,  8, -1 },
    { -1, -1, -1 }
}};


// Structuring elements

// Cross-shaped
constexpr Kernel2D<3,3,float> Cross {{
    { 0, 1, 0 },
    { 1, 1, 1 },
    { 0, 1, 0 }
}};

// Diamond
constexpr Kernel2D<5,5,float> Diamond {{
    { 0, 0, 1, 0, 0 },
    { 0, 1, 1, 1, 0 },
    { 1, 1, 1, 1, 1 },
    { 0, 1, 1, 1, 0 },
    { 0, 0, 1, 0, 0 }
}};

// Kirsch compass (one direction)
constexpr Kernel2D<3,3,float> KirschN {{
    {  5,  5,  5 },
    { -3,  0, -3 },
    { -3, -3, -3 }
}};

// Roberts cross (X)
constexpr Kernel2D<2,2,float> RobertsX {{
    {  1,  0 },
    {  0, -1 }
}};

// Roberts cross (Y)
constexpr Kernel2D<2,2,float> RobertsY {{
    {  0,  1 },
    { -1,  0 }
}};

