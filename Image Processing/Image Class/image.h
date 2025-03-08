#ifndef IMAGE_H
#define IMAGE_H

class Image{
    public:
        Image();
        Image(int num_rows, int num_cols, int depth);
        ~Image();

        int get_pixel_val (int row, int col);
        void set_pixel_val (int row, int col, int value);
        int get_pixel_val_ch(int row, int col, int depth);
        void set_pixel_val_ch (int row, int col, int depth, int value);
    
    private:
        int rows;
        int columns;
        int depth;
        int **pixel_val;
        int ***pixel_val_ch;
};

#endif // IMAGE_H