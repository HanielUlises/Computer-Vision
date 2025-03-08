#include "image.h"

#include <stdlib.h>

Image::Image()
{
    rows = 0;
    columns = 0;
    depth = 0;
    pixel_val = NULL;
    pixel_val_ch = NULL;
}

Image::Image(int num_rows, int num_columns, int n_depth)
{
    rows = num_rows;
    columns = num_columns;
    depth = n_depth;
    pixel_val = new int *[rows];
    pixel_val_ch = new int **[rows];

    for (int i = 0; i < rows; i++)
    {
        pixel_val[i] = new int[columns];
        for (int j = 0; j < columns; j++)
        {
            pixel_val[i][j] = 0;
        }
    }

    for (int i = 0; i < rows; i++)
    {
        pixel_val_ch[i] = new int *[columns];
        for (int j = 0; j < columns; j++)
        {
            pixel_val_ch[i][j] = new int[depth];
            for (int k = 0; k < depth; k++)
            {
                pixel_val_ch[i][j][k] = 0;
            }
        }
    }
}

Image::~Image()
{
    for (int i = 0; i < rows; i++)
        delete pixel_val[i];

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            delete pixel_val_ch[i][j];
        }
        delete pixel_val_ch[i];
    }
    rows = 0;
    columns = 0;
    depth = 0;

    delete pixel_val;
    delete pixel_val_ch;
}

int Image::get_pixel_val(int row, int col){
    return pixel_val[row][col];
}

int Image::get_pixel_val_ch(int row, int col, int depth){
    return pixel_val_ch[row][col][depth];
}

void Image::set_pixel_val(int row, int col, int value){
    pixel_val[row][col] = value;
}

void Image::set_pixel_val_ch(int row, int col, int depth, int value){
    pixel_val_ch[row][col][depth] = value;
}