
#include <stdio.h>

//void timer_start(int i);

//double timer_stop(int i);


extern size_t H;
extern size_t W;
extern size_t SIZE;

void alloc_mat(float **in, int H, int W);
void set_zeros(float *in, int H, int W);
void set_ones(float *in, int H, int W);
float max(float *in, int n);
void clip(float* in, int start, int end);
void find_zero(float* in, float* out);
void cumsum(float* in, float* out, int axis);
void diff(float *in, float* out, int axis);

void replace_new(float* in, float _old, float _new);
void replace_th(float* in, float th);

void abs_mat(float* in);
void add_val(float *A, float v, float* C);
void add_mat(float *A, float *B, float* C);
void sub_mat(float *A, float *B, float* C);
void div_mat(float *A, float *B, float* C);
void mul_mat(float *A, float *B, float* C);
void mul_mat_inv(float *A, float *B, float* C);

static double get_time();
void timer_start(int i);
double timer_stop(int i);

//bool write_file(const char* fn, size_t sz, void* buf);
//bool write_file2(const float* fn, size_t sz, void* buf);
