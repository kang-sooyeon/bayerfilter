#include <CL/cl.h>


// main
void init_resources();
void read_rawfile(const char* filePath);
void read_gpu_buffer();
void demosaic();
void write_gpu_buffer();
void save_png(const char* filePath);
void free_resources();

// demosaic method
void green_interpolation();
void red_blue_interpolation();
double get_time();
void timer_start(int i);
double timer_stop(int i);
void print_device_info(cl_device_id device);
cl_program create_and_build_program_with_source(cl_context context, cl_device_id device, const char *file_name);

// validation method
void print_kernel_time();
void validation();
int diffrgb_float(const char*fn1, const char* fn2);
int diffrgb(const char*fn1, const char* fn2);
void write_file(const char* filePath);
void write_file_float(const char* filePath);
void* read_file(const char *fn, size_t *sz);


