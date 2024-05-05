#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>

// Histogram operations
void get_histogram();   // Compute and display histogram of an image
void get_LUT();         // Generate and apply a Look-Up Table

// Color space manipulation
void change_color_space(); // Change the color space of an image

// Mathematical operations on images
void addition();        // Perform addition of two images
void subtraction();     // Perform subtraction between two images
void multiplication();  // Perform multiplication of two images
void division();        // Perform division between two images

// Logical operations on images
void and_img();         // Perform AND operation between two images
void or_img();          // Perform OR operation between two images
void xor_img();         // Perform XOR operation between two images

// Correlation and convolution
void convolution();     // Apply a convolution operation on an image
void correlation();     // Apply a correlation operation on an image

// Geometrical operations
void geometrical_operations();

#endif // IMAGE_PROCESSING_H
