#include "util.h"

void init();
void read_rawfile(const char* filePath);
void write_file(const char* filePath);
void save_png(const char* filePath);
void demosaic();
void set_bayer_mask();

void imfilter1D_h(float* in, float* filter, int R, float* out);
void imfilter1D_v(float* in, float* filter, int R, float* out);
void imfilter1D_h_replicate(float* in, float* filter, int R, float* out);
void imfilter1D_v_replicate(float* in, float* filter, int R, float* out);
void imfilter2D(float* in, float* filter, int R, int S, float* out);
void imfilter2D_replicate(float* in, float* filter, int R, int S, float* out);
void boxfilter(float* in, int h, int v, float *out);
void guidedfilter(float* I, float* p, float* M, int h, int v, float* out);

void red_interpolation();
void green_interpolation();
void blue_interpolation();

void free_resources();
