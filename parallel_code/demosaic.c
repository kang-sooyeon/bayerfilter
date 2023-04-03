
#include "demosaic.h"
#include <stdio.h>
#include <CL/cl.h>
#include "svpng.inc"
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#include "util.h"
#include <sys/time.h>

//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////// Options ///
//#define PRINT_KERNEL_TIME
#define USE_HALF_PRECISON
//////////////////////////////////////////////////////////////////////

#define C 3
size_t H = 3496;
size_t W = 4656;

double start_time[8];
// string pattern = "grbg";
// bayer filter pattern : "grbg"

#ifdef USE_HALF_PRECISON
#define DATA_TYPE cl_half
#else
#define DATA_TYPE float
#endif

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

double total_time = 0;
cl_ulong time_start;
cl_ulong time_end;

static cl_int err;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_event event;

// green
cl_kernel kernel_imfilter2;
cl_kernel kernel_linear_coefficient_h;
cl_kernel kernel_linear_coefficient_v;
cl_kernel kernel_tentative_residual_h;
cl_kernel kernel_tentative_residual_v;
cl_kernel kernel_residual_interpolation;
cl_kernel kernel_directional_weight;
cl_kernel kernel_compute_weight_new;
// red_blue
cl_kernel kernel_red_blue_vertical_sum;
cl_kernel kernel_horizontal_sum_and_linear_coefficient;
cl_kernel kernel_coefficient_vertical_sum;
cl_kernel kernel_horizontal_sum_and_tentative_residual;
cl_kernel kernel_red_blue_residual_interpolation_and_add_tentative;
cl_kernel kernel_color_difference_h;
cl_kernel kernel_color_difference_v;

#ifdef USE_HALF_PRECISON
cl_kernel kernel_convert_to_fp16;
cl_kernel kernel_convert_from_fp16;
#endif

float* input_rgb;
float* output_green;      // interpolated green : H * W
float* output_red;        // interpolated red : H * W
float* output_blue;       // interpolated blue : H * W

#ifdef USE_HALF_PRECISON
cl_mem raw_d;
cl_mem raw_H_d;
cl_mem tentative_H_d;
cl_mem tentative_V_d;
#else
float* raw_d;
float* raw_H_d;
float* tentative_H_d;
float* tentative_V_d;
#endif

cl_mem raw_V_d;
cl_mem A_H_d, B_H_d, C_H_d, D_H_d, A_V_d, B_V_d, C_V_d, D_V_d;
cl_mem residual_H_d, residual_V_d;
cl_mem dif_H_d, dif_V_d;

void print_platform_info(cl_platform_id platform);
void print_device_info(cl_device_id device);
cl_program create_and_build_program_with_source(
    cl_context context, cl_device_id device, const char *file_name);


double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) {
  start_time[i] = get_time();
}

double timer_stop(int i) {
  return get_time() - start_time[i];
}


void write_file_float(const char* filePath) {
  timer_start(0);
  // result
  float *result = (float*)malloc(H*W*C*sizeof(float));
  for (int i = 0; i < H * W; i++ ) {
    result[i*3+0] = output_red[i] * 8.0;
    result[i*3+1] = output_green[i] * 8.0;
    result[i*3+2] = output_blue[i] * 8.0;
  }

  FILE* out = NULL;
  out = fopen(filePath, "wb");
  if (out == NULL) {
    printf("write_file open error");
    return;
  }

  fwrite(result, sizeof(float), H * W * C, out);

  fclose(out);
  free(result);

  printf("write_file() complete : %lf\n", timer_stop(0));
}

void write_file(const char* filePath) {
  timer_start(0);
  // result
  unsigned char *result = (unsigned char*)malloc(H*W*C*sizeof(unsigned char));
  for (int i = 0; i < H * W; i++ ) {
    result[i*3+0] = (unsigned char)(output_red[i] * 8.0);
    result[i*3+1] = (unsigned char)(output_green[i] * 8.0);
    result[i*3+2] = (unsigned char)(output_blue[i] * 8.0);
  }

  FILE* out = NULL;
  out = fopen(filePath, "wb");
  if (out == NULL) {
    printf("write_file open error");
    return;
  }

  fwrite(result, sizeof(unsigned char), H * W * C, out);

  fclose(out);
  free(result);

  printf("write_file() complete : %lf\n", timer_stop(0));
}


void save_png(const char* filePath) {
  timer_start(0);
  unsigned char *result = (unsigned char*)malloc(H*W*C*sizeof(unsigned char));
  for (int i = 0; i < H * W; i++ ) {
    result[i*3+0] = (unsigned char)(output_red[i] * 8.0);
    result[i*3+1] = (unsigned char)(output_green[i] * 8.0);
    result[i*3+2] = (unsigned char)(output_blue[i] * 8.0);
  }

  FILE* out = NULL;
  out = fopen(filePath, "wb");
  if (out == NULL) {
    printf("write_file open error");
    return;
  }

  svpng(out, W, H, result, 0);
  fclose(out);
  free(result);

  printf("write_png() complete : %lf\n", timer_stop(0));
}

void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);

  cl_ulong global_mem_size;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(cl_ulong), &global_mem_size, NULL));
  printf("global mem size : %lu (byte)\n", global_mem_size);

  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(cl_ulong), &global_mem_size, NULL));
  printf("max mem alloc size : %lu (byte)\n", global_mem_size);

  size_t max_work_item_size[3]; 
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizeof(max_work_item_size), &max_work_item_size, NULL));
  printf("DEVICE_MAX_WORK_ITEM_SIZE = (%lu,%lu,%lu)\n",
      max_work_item_size[0], max_work_item_size[1], max_work_item_size[2]);

  size_t max_work_group_size; 
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(max_work_group_size), &max_work_group_size, NULL));
  printf("DEVICE_MAX_WORK_GROUP_SIZE = %lu\n",
      max_work_group_size);

  cl_uint comp_units;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cl_uint), &comp_units, NULL));
  printf("compute units : %d\n", comp_units);
}

cl_program create_and_build_program_with_source(
    cl_context context, cl_device_id device, const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char*)malloc(source_size + 1);
  fread(source_code, sizeof(char), source_size, file);
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(context, 1,
      (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
    char *log = (char*)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }

  CHECK_ERROR(err);
  return program;
}

void print_kernel_time(char* kernel_name) {
  printf("%s : ", kernel_name);
  err = clWaitForEvents(1, &event);
  CHECK_ERROR(err);

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
      sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
      sizeof(time_end), &time_end, NULL);

  double nanoSeconds = time_end - time_start;
  printf(" %0.3f ms\n", nanoSeconds / 1000000.0);
}

void init_resources() {
  timer_start(0);  

  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);
  //print_platform_info(platform);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);
  //print_device_info(device);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

#ifdef PRINT_KERNEL_TIME
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#else
  queue = clCreateCommandQueue(context, device, 0,  &err);
#endif
  CHECK_ERROR(err);

  ////////////////////////////////////// kernel_green.cl
#ifdef USE_HALF_PRECISON
  program = create_and_build_program_with_source(context, device, "kernel_green_fp16.cl");
#else
  program = create_and_build_program_with_source(context, device, "kernel_green.cl");
#endif
  kernel_imfilter2 = clCreateKernel(program, "imfilter2", &err);
  CHECK_ERROR(err);
  kernel_residual_interpolation = clCreateKernel(program, "residual_interpolation", &err);
  CHECK_ERROR(err);
  kernel_compute_weight_new = clCreateKernel(program, "compute_weight_new", &err);
  CHECK_ERROR(err);
  kernel_directional_weight = clCreateKernel(program, "directional_weight", &err);
  CHECK_ERROR(err);
  kernel_linear_coefficient_h = clCreateKernel(program, "linear_coefficient_horizontal", &err);
  CHECK_ERROR(err);
  kernel_linear_coefficient_v = clCreateKernel(program, "linear_coefficient_vertical", &err);
  CHECK_ERROR(err);
  kernel_tentative_residual_h = clCreateKernel(program, "tentative_residual_horizontal", &err);
  CHECK_ERROR(err);
  kernel_tentative_residual_v = clCreateKernel(program, "tentative_residual_vertical", &err);
  CHECK_ERROR(err);
  kernel_color_difference_h = clCreateKernel(program, "color_difference_h", &err);
  CHECK_ERROR(err);
  kernel_color_difference_v = clCreateKernel(program, "color_difference_v", &err);
  CHECK_ERROR(err);

  ////////////////////////////////////// kernel_red_blue.cl
#ifdef USE_HALF_PRECISON
  program = create_and_build_program_with_source(context, device, "kernel_red_blue_fp16.cl");
#else
  program = create_and_build_program_with_source(context, device, "kernel_red_blue.cl");
#endif
  kernel_red_blue_vertical_sum =
    clCreateKernel(program, "red_blue_vertical_sum", &err);
  CHECK_ERROR(err);
  kernel_horizontal_sum_and_linear_coefficient =
    clCreateKernel(program, "horizontal_sum_and_linear_coefficient", &err);
  CHECK_ERROR(err);
  kernel_coefficient_vertical_sum =
    clCreateKernel(program, "coefficient_vertical_sum", &err);
  CHECK_ERROR(err);
  kernel_horizontal_sum_and_tentative_residual =
    clCreateKernel(program, "horizontal_sum_and_tentative_residual", &err);
  CHECK_ERROR(err);
  kernel_red_blue_residual_interpolation_and_add_tentative =
    clCreateKernel(program, "red_blue_residual_interpolation_and_add_tentative", &err);
  CHECK_ERROR(err);

#ifdef USE_HALF_PRECISON
  program = create_and_build_program_with_source(context, device, "kernel_fp16.cl");
  kernel_convert_to_fp16 = clCreateKernel(program, "convert_to_fp16", &err);
  CHECK_ERROR(err);
  kernel_convert_from_fp16 = clCreateKernel(program, "convert_from_fp16", &err);
  CHECK_ERROR(err);
#endif

  input_rgb = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE,
      H * W * sizeof(float), 0);
  clEnqueueSVMMap(queue, CL_NON_BLOCKING, CL_MAP_WRITE, input_rgb,
      H * W * sizeof(float), 0, NULL, &event);
  err = clWaitForEvents(1, &event);

#ifdef USE_HALF_PRECISON
  raw_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
      H * W * sizeof(cl_half), NULL, &err);
  CHECK_ERROR(err);

  raw_H_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
      H * W * sizeof(cl_half), NULL, &err);
  CHECK_ERROR(err);

  tentative_H_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
      H * W * sizeof(cl_half), NULL, &err);
  CHECK_ERROR(err);

  tentative_V_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
      H * W * sizeof(cl_half), NULL, &err);
  CHECK_ERROR(err);

  output_green = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE,
      H * W * sizeof(float), 0);
  clEnqueueSVMMap(queue, CL_NON_BLOCKING, CL_MAP_WRITE, output_green,
      H * W * sizeof(float), 0, NULL, &event);
  err = clWaitForEvents(1, &event);
  
  output_red = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE,
      H * W * sizeof(float), 0);
  clEnqueueSVMMap(queue, CL_NON_BLOCKING, CL_MAP_WRITE, output_red,
      H * W * sizeof(float), 0, NULL, &event);
  err = clWaitForEvents(1, &event);

  output_blue = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE,
      H * W * sizeof(float), 0);
  clEnqueueSVMMap(queue, CL_NON_BLOCKING, CL_MAP_WRITE, output_blue,
      H * W * sizeof(float), 0, NULL, &event);
  err = clWaitForEvents(1, &event);
#else
  raw_H_d = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE,
      H * W * sizeof(float), 0);
  clEnqueueSVMMap(queue, CL_NON_BLOCKING, CL_MAP_WRITE, raw_H_d,
      H * W * sizeof(float), 0, NULL, &event);
  err = clWaitForEvents(1, &event);
  
  tentative_H_d = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE,
      H * W * sizeof(float), 0);
  clEnqueueSVMMap(queue, CL_NON_BLOCKING, CL_MAP_WRITE, tentative_H_d,
      H * W * sizeof(float), 0, NULL, &event);
  err = clWaitForEvents(1, &event);

  tentative_V_d = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE,
      H * W * sizeof(float), 0);
  clEnqueueSVMMap(queue, CL_NON_BLOCKING, CL_MAP_WRITE, tentative_V_d,
      H * W * sizeof(float), 0, NULL, &event);
  err = clWaitForEvents(1, &event);
#endif
  
  raw_V_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);

  A_H_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);
  B_H_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);
  C_H_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);
  D_H_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);

  A_V_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);
  B_V_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);
  C_V_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);
  D_V_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);

  residual_H_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);
  residual_V_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * sizeof(DATA_TYPE), NULL, &err);
  CHECK_ERROR(err);

  clFinish(queue);
  printf("init() complete : %lf sec\n", timer_stop(0));
}

void read_rawfile(const char* filePath) {
  timer_start(0);

  FILE* in = NULL;
  in = fopen(filePath, "rb");

  if (in == NULL) {
    printf("read_file open error");
    return;
  }
  short* img_buf = (short*)malloc(H * W * sizeof(short));
  fread(img_buf, sizeof(short), H * W, in);
  fclose(in);

  for (int i = 0; i < H * W; i++) {
    input_rgb[i] = ((float)img_buf[i] / 1024.0) * 32.0;
  }
  free(img_buf);  

  printf("read_rawfile() complete : %lf sec\n", timer_stop(0));
}

void write_gpu_buffer() {
  timer_start(0);

  clEnqueueSVMUnmap(queue, input_rgb, 0, NULL, &event);

  clFinish(queue);
  printf("write_gpu_buffer() complete : %lf sec\n", timer_stop(0));
}

#ifdef USE_HALF_PRECISON
void linear_coefficient_horizontal(cl_mem P, cl_mem I_H,
    cl_mem A_H_d, cl_mem B_H_d, cl_mem C_H_d, cl_mem D_H_d) {
  err = clSetKernelArg(kernel_linear_coefficient_h, 0, sizeof(cl_mem), &P);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_h, 1, sizeof(cl_mem), &I_H);
  CHECK_ERROR(err);
#else
void linear_coefficient_horizontal(float* P, float* I_H,
    cl_mem A_H_d, cl_mem B_H_d, cl_mem C_H_d, cl_mem D_H_d) {
  err = clSetKernelArgSVMPointer(kernel_linear_coefficient_h, 0, P);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel_linear_coefficient_h, 1, I_H);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_linear_coefficient_h, 2, sizeof(cl_mem), &A_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_h, 3, sizeof(cl_mem), &B_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_h, 4, sizeof(cl_mem), &C_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_h, 5, sizeof(cl_mem), &D_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_h, 6, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_h, 7, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_linear_coefficient_h, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("linear_coefficient_h()");
#endif
}

#ifdef USE_HALF_PRECISON
void linear_coefficient_vertical(cl_mem P, cl_mem I_V_d,
    cl_mem A_V_d, cl_mem B_V_d, cl_mem C_V_d, cl_mem D_V_d) {
  err = clSetKernelArg(kernel_linear_coefficient_v, 0, sizeof(cl_mem), &P);
  CHECK_ERROR(err);
#else
void linear_coefficient_vertical(float* P, cl_mem I_V_d,
    cl_mem A_V_d, cl_mem B_V_d, cl_mem C_V_d, cl_mem D_V_d) {
  err = clSetKernelArgSVMPointer(kernel_linear_coefficient_v, 0, P);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_linear_coefficient_v, 1, sizeof(cl_mem), &I_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_v, 2, sizeof(cl_mem), &A_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_v, 3, sizeof(cl_mem), &B_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_v, 4, sizeof(cl_mem), &C_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_v, 5, sizeof(cl_mem), &D_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_v, 6, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_linear_coefficient_v, 7, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_1d, lws_1d;
  gws_1d = W / 2;
  lws_1d = 48;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;
  err = clEnqueueNDRangeKernel(queue, kernel_linear_coefficient_v, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("linear_coefficient_v()");
#endif
}

#ifdef USE_HALF_PRECISON
void tentative_residual_horizontal(cl_mem P, cl_mem I_H,
    cl_mem A_H_d, cl_mem B_H_d, cl_mem C_H_d, cl_mem D_H_d,
    cl_mem tentative_H, cl_mem residual_H_d) {
  err = clSetKernelArg(kernel_tentative_residual_h, 0, sizeof(cl_mem), &P);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_h, 1, sizeof(cl_mem), &I_H);
  CHECK_ERROR(err);
#else
void tentative_residual_horizontal(float* P, float* I_H,
    cl_mem A_H_d, cl_mem B_H_d, cl_mem C_H_d, cl_mem D_H_d,
    float* tentative_H, cl_mem residual_H_d) {
  err = clSetKernelArgSVMPointer(kernel_tentative_residual_h, 0, P);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel_tentative_residual_h, 1, I_H);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_tentative_residual_h, 2, sizeof(cl_mem), &A_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_h, 3, sizeof(cl_mem), &B_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_h, 4, sizeof(cl_mem), &C_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_h, 5, sizeof(cl_mem), &D_H_d);
  CHECK_ERROR(err);
#ifdef USE_HALF_PRECISON
  err = clSetKernelArg(kernel_tentative_residual_h, 6, sizeof(cl_mem), &tentative_H);
  CHECK_ERROR(err);
#else
  err = clSetKernelArgSVMPointer(kernel_tentative_residual_h, 6, tentative_H);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_tentative_residual_h, 7, sizeof(cl_mem), &residual_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_h, 8, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_h, 9, sizeof(int), &W);
  CHECK_ERROR(err);
  
  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_tentative_residual_h, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("tentative_residual_horizontal()");
#endif
}

#ifdef USE_HALF_PRECISON
void tentative_residual_vertical(cl_mem P_d, cl_mem I_V_d,
    cl_mem A_V_d, cl_mem B_V_d, cl_mem C_V_d, cl_mem D_V_d,
    cl_mem tentative_V, cl_mem residual_V_d) {
  err = clSetKernelArg(kernel_tentative_residual_v, 0, sizeof(cl_mem), &P_d);
  CHECK_ERROR(err);
#else
void tentative_residual_vertical(float* P_d, cl_mem I_V_d,
    cl_mem A_V_d, cl_mem B_V_d, cl_mem C_V_d, cl_mem D_V_d,
    float* tentative_V, cl_mem residual_V_d) {
  err = clSetKernelArgSVMPointer(kernel_tentative_residual_v, 0, P_d);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_tentative_residual_v, 1, sizeof(cl_mem), &I_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_v, 2, sizeof(cl_mem), &A_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_v, 3, sizeof(cl_mem), &B_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_v, 4, sizeof(cl_mem), &C_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_v, 5, sizeof(cl_mem), &D_V_d);
  CHECK_ERROR(err);
#ifdef USE_HALF_PRECISON
  err = clSetKernelArg(kernel_tentative_residual_v, 6, sizeof(cl_mem), &tentative_V);
  CHECK_ERROR(err);
#else
  err = clSetKernelArgSVMPointer(kernel_tentative_residual_v, 6, tentative_V);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_tentative_residual_v, 7, sizeof(cl_mem), &residual_V_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_v, 8, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_tentative_residual_v, 9, sizeof(int), &W);
  CHECK_ERROR(err);
  
  size_t gws_1d, lws_1d;
  gws_1d = W / 2;
#ifdef USE_HALF_PRECISON
  lws_1d = 48;
#else
  lws_1d = 384;
#endif
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;
  err = clEnqueueNDRangeKernel(queue, kernel_tentative_residual_v, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("tentative_residual_v()");
#endif
}

#ifdef USE_HALF_PRECISON
void residual_interpolation_and_add_tentative(cl_mem raw_d,
    cl_mem residual_H_d, cl_mem residual_V_d,
    cl_mem tentative_H, cl_mem tentative_V) {
  err = clSetKernelArg(kernel_residual_interpolation, 0, sizeof(cl_mem), &raw_d);
  CHECK_ERROR(err);
#else
void residual_interpolation_and_add_tentative(float* raw_d,
    cl_mem residual_H_d, cl_mem residual_V_d,
    float* tentative_H, float* tentative_V) {
  err = clSetKernelArgSVMPointer(kernel_residual_interpolation, 0, raw_d);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_residual_interpolation, 1, sizeof(cl_mem), &residual_H_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_residual_interpolation, 2, sizeof(cl_mem), &residual_V_d);
  CHECK_ERROR(err);
#ifdef USE_HALF_PRECISON
  err = clSetKernelArg(kernel_residual_interpolation, 3, sizeof(cl_mem), &tentative_H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_residual_interpolation, 4, sizeof(cl_mem), &tentative_V);
  CHECK_ERROR(err);
#else
  err = clSetKernelArgSVMPointer(kernel_residual_interpolation, 3, tentative_H);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel_residual_interpolation, 4, tentative_V);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_residual_interpolation, 5, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_residual_interpolation, 6, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;


  err = clEnqueueNDRangeKernel(queue, kernel_residual_interpolation, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("residual_interpolation()");
#endif
}

#ifdef USE_HALF_PRECISON
void color_difference_h(cl_mem difh, cl_mem difh2_d, cl_mem difw_d, cl_mem dife_d) {
  err = clSetKernelArg(kernel_color_difference_h, 0, sizeof(cl_mem), &difh);
  CHECK_ERROR(err);
#else
void color_difference_h(float* difh, cl_mem difh2_d, cl_mem difw_d, cl_mem dife_d) {
  err = clSetKernelArgSVMPointer(kernel_color_difference_h, 0, difh);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_color_difference_h, 1, sizeof(cl_mem), &difh2_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_color_difference_h, 2, sizeof(cl_mem), &difw_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_color_difference_h, 3, sizeof(cl_mem), &dife_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_color_difference_h, 4, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_color_difference_h, 5, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_color_difference_h, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("color_difference_h()");
#endif
}

#ifdef USE_HALF_PRECISON
void color_difference_v(cl_mem difv, cl_mem difv2_d, cl_mem difn_d, cl_mem difs_d) {
  err = clSetKernelArg(kernel_color_difference_v, 0, sizeof(cl_mem), &difv);
  CHECK_ERROR(err);
#else
void color_difference_v(float* difv, cl_mem difv2_d, cl_mem difn_d, cl_mem difs_d) {
  err = clSetKernelArgSVMPointer(kernel_color_difference_v, 0, difv);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_color_difference_v, 1, sizeof(cl_mem), &difv2_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_color_difference_v, 2, sizeof(cl_mem), &difn_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_color_difference_v, 3, sizeof(cl_mem), &difs_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_color_difference_v, 4, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_color_difference_v, 5, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_2d[2], lws_2d[2];
  gws_2d[0] = W; gws_2d[1] = H;
  lws_2d[0] = 4; lws_2d[1] = 6; 
  for (int i = 0; i < 2; ++i) {
    gws_2d[i] = (gws_2d[i] + lws_2d[i] - 1) / lws_2d[i] * lws_2d[i];
  }

  err = clEnqueueNDRangeKernel(queue, kernel_color_difference_v, 2, NULL,
      gws_2d, lws_2d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("color_difference_v()");
#endif
}

void compute_weight_new(cl_mem in_d, cl_mem out_d) {
  err = clSetKernelArg(kernel_compute_weight_new, 0, sizeof(cl_mem), &in_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_compute_weight_new, 1, sizeof(cl_mem), &out_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_compute_weight_new, 2, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_compute_weight_new, 3, sizeof(int), &W);
  CHECK_ERROR(err);
 
  size_t gws_2d[2], lws_2d[2];
  gws_2d[0] = W/2; gws_2d[1] = H/2;
  lws_2d[0] = 4; lws_2d[1] = 6; 
  for (int i = 0; i < 2; ++i) {
    gws_2d[i] = (gws_2d[i] + lws_2d[i] - 1) / lws_2d[i] * lws_2d[i];
  }

  err = clEnqueueNDRangeKernel(queue, kernel_compute_weight_new, 2, NULL,
      gws_2d, lws_2d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("compute_weight_new()");
#endif
}

#ifdef USE_HALF_PRECISON
void directional_weight(cl_mem raw, cl_mem difn_d, cl_mem difs_d,
    cl_mem difw_d, cl_mem dife_d, cl_mem wh_d, cl_mem wv_d, cl_mem green) {
  err = clSetKernelArg(kernel_directional_weight, 0, sizeof(cl_mem), &raw);
  CHECK_ERROR(err);
#else
void directional_weight(float* raw, cl_mem difn_d, cl_mem difs_d,
    cl_mem difw_d, cl_mem dife_d, cl_mem wh_d, cl_mem wv_d, float* green) {
  err = clSetKernelArgSVMPointer(kernel_directional_weight, 0, raw);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_directional_weight, 1, sizeof(cl_mem), &difn_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_directional_weight, 2, sizeof(cl_mem), &difs_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_directional_weight, 3, sizeof(cl_mem), &difw_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_directional_weight, 4, sizeof(cl_mem), &dife_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_directional_weight, 5, sizeof(cl_mem), &wh_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_directional_weight, 6, sizeof(cl_mem), &wv_d);
  CHECK_ERROR(err);
#ifdef USE_HALF_PRECISON
  err = clSetKernelArg(kernel_directional_weight, 7, sizeof(cl_mem), &green);
  CHECK_ERROR(err);
#else
  err = clSetKernelArgSVMPointer(kernel_directional_weight, 7, green);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_directional_weight, 8, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_directional_weight, 9, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_directional_weight, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("directional_weight()");
#endif
}

#ifdef USE_HALF_PRECISON
void red_blue_vertical_sum(cl_mem I, cl_mem P,
    cl_mem sumI_d, cl_mem sumP_d, cl_mem sumII_d, cl_mem sumIP_d) {
  err = clSetKernelArg(kernel_red_blue_vertical_sum, 0, sizeof(cl_mem), &I);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_vertical_sum, 1, sizeof(cl_mem), &P);
  CHECK_ERROR(err);
#else
void red_blue_vertical_sum(float* I, float* P,
    cl_mem sumI_d, cl_mem sumP_d, cl_mem sumII_d, cl_mem sumIP_d) {
  err = clSetKernelArgSVMPointer(kernel_red_blue_vertical_sum, 0, I);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel_red_blue_vertical_sum, 1, P);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_red_blue_vertical_sum, 2, sizeof(cl_mem), &sumI_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_vertical_sum, 3, sizeof(cl_mem), &sumP_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_vertical_sum, 4, sizeof(cl_mem), &sumII_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_vertical_sum, 5, sizeof(cl_mem), &sumIP_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_vertical_sum, 6, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_vertical_sum, 7, sizeof(int), &W);
  CHECK_ERROR(err);
  
  size_t gws_1d, lws_1d;
  gws_1d = W / 2;
#ifdef USE_HALF_PRECISON
  lws_1d = 48;
#else
  lws_1d = 24;
#endif

  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_red_blue_vertical_sum, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("red_blue_vertical_sum()");
#endif
}

void horizontal_sum_and_linear_coefficient(cl_mem sumI_d, cl_mem sumP_d,
    cl_mem sumII_d, cl_mem sumIP_d,
    cl_mem coefficientA_5_d, cl_mem coefficientB_5_d,
    cl_mem coefficientA_6_d, cl_mem coefficientB_6_d) {
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 0, sizeof(cl_mem), &sumI_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 1, sizeof(cl_mem), &sumP_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 2, sizeof(cl_mem), &sumII_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 3, sizeof(cl_mem), &sumIP_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 4, sizeof(cl_mem), &coefficientA_5_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 5, sizeof(cl_mem), &coefficientB_5_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 6, sizeof(cl_mem), &coefficientA_6_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 7, sizeof(cl_mem), &coefficientB_6_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 8, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_linear_coefficient, 9, sizeof(int), &W);
  CHECK_ERROR(err);
  
  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_horizontal_sum_and_linear_coefficient, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("kernel_horizontal_sum_and_linear_coefficient()");
#endif
}

void coefficient_vertical_sum(cl_mem in_d, cl_mem out_d) {
  err = clSetKernelArg(kernel_coefficient_vertical_sum, 0, sizeof(cl_mem), &in_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_coefficient_vertical_sum, 1, sizeof(cl_mem), &out_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_coefficient_vertical_sum, 2, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_coefficient_vertical_sum, 3, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_1d, lws_1d;
  gws_1d = H * W / 2;
#ifdef USE_HALF_PRECISON
  lws_1d = 48;
#else
  lws_1d = 384;
#endif

  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_coefficient_vertical_sum, 1, NULL, &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("kernel_coefficient_vertical_sum()");
#endif
}

#ifdef USE_HALF_PRECISON
void horizontal_sum_and_tentative_residual(cl_mem green, cl_mem raw,
    cl_mem coefficientA_d, cl_mem coefficientB_d,
    cl_mem coefficientA_INV_d, cl_mem coefficientB_INV_d,
    cl_mem tentativeR, cl_mem tentativeB, cl_mem residualRB_d) {
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 0, sizeof(cl_mem), &green);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 1, sizeof(cl_mem), &raw);
  CHECK_ERROR(err);
#else
void horizontal_sum_and_tentative_residual(float* green, float* raw,
    cl_mem coefficientA_d, cl_mem coefficientB_d,
    cl_mem coefficientA_INV_d, cl_mem coefficientB_INV_d,
    float* tentativeR, float* tentativeB, cl_mem residualRB_d) {
  err = clSetKernelArgSVMPointer(kernel_horizontal_sum_and_tentative_residual, 0, green);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel_horizontal_sum_and_tentative_residual, 1, raw);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 2, sizeof(cl_mem), &coefficientA_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 3, sizeof(cl_mem), &coefficientB_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 4, sizeof(cl_mem), &coefficientA_INV_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 5, sizeof(cl_mem), &coefficientB_INV_d);
  CHECK_ERROR(err);
#ifdef USE_HALF_PRECISON
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 6, sizeof(cl_mem), &tentativeR);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 7, sizeof(cl_mem), &tentativeB);
  CHECK_ERROR(err);
#else
  err = clSetKernelArgSVMPointer(kernel_horizontal_sum_and_tentative_residual, 6, tentativeR);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel_horizontal_sum_and_tentative_residual, 7, tentativeB);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 8, sizeof(cl_mem), &residualRB_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 9, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_horizontal_sum_and_tentative_residual, 10, sizeof(int), &W);
  CHECK_ERROR(err);
  
  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue,
      kernel_horizontal_sum_and_tentative_residual, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("kernel_horizontal_sum_and_tentative_residual()");
#endif
}

#ifdef USE_HALF_PRECISON
void red_blue_residual_interpolation_and_add_tentative(cl_mem residualRB_d,
    cl_mem tentativeR, cl_mem tentativeB) {
  err = clSetKernelArg(kernel_red_blue_residual_interpolation_and_add_tentative,
      0, sizeof(cl_mem), &residualRB_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_residual_interpolation_and_add_tentative,
      1, sizeof(cl_mem), &tentativeR);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_residual_interpolation_and_add_tentative,
      2, sizeof(cl_mem), &tentativeB);
  CHECK_ERROR(err);
#else
void red_blue_residual_interpolation_and_add_tentative(cl_mem residualRB_d,
    float* tentativeR, float* tentativeB) {
  err = clSetKernelArg(kernel_red_blue_residual_interpolation_and_add_tentative,
      0, sizeof(cl_mem), &residualRB_d);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(
      kernel_red_blue_residual_interpolation_and_add_tentative, 1, tentativeR);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(
      kernel_red_blue_residual_interpolation_and_add_tentative, 2, tentativeB);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_red_blue_residual_interpolation_and_add_tentative,
      3, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_red_blue_residual_interpolation_and_add_tentative,
      4, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;


  err = clEnqueueNDRangeKernel(queue,
      kernel_red_blue_residual_interpolation_and_add_tentative, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("kernel_red_blue_residual_interpolation_and_add_tentative()");
#endif
}

#ifdef USE_HALF_PRECISON
void imfilter2(cl_mem in, cl_mem out1, cl_mem out2_d) {
  err = clSetKernelArg(kernel_imfilter2, 0, sizeof(cl_mem), &in);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_imfilter2, 1, sizeof(cl_mem), &out1);
  CHECK_ERROR(err);
#else
void imfilter2(float* in, float* out1, cl_mem out2_d) {
  err = clSetKernelArgSVMPointer(kernel_imfilter2, 0, in);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel_imfilter2, 1, out1);
  CHECK_ERROR(err);
#endif
  err = clSetKernelArg(kernel_imfilter2, 2, sizeof(cl_mem), &out2_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_imfilter2, 3, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_imfilter2, 4, sizeof(int), &W);
  CHECK_ERROR(err);

  size_t gws_1d, lws_1d;
  gws_1d = H * W;
#ifdef USE_HALF_PRECISON
  lws_1d = 48;
#else
  lws_1d = 384;
#endif

  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_imfilter2, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef PRINT_KERNEL_TIME
  print_kernel_time("imfilter2()");
#endif
}

#ifdef USE_HALF_PRECISON
void convert_to_fp16(float* in, cl_mem out_d) {
  err = clSetKernelArgSVMPointer(kernel_convert_to_fp16, 0, in);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convert_to_fp16, 1, sizeof(cl_mem), &out_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convert_to_fp16, 2, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convert_to_fp16, 3, sizeof(int), &W);
  CHECK_ERROR(err);
  
  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_convert_to_fp16, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef kernel_time
  print_kernel_time("convert_to_fp16()");
#endif
}

void convert_from_fp16(cl_mem in_d, float* out) {
  err = clSetKernelArg(kernel_convert_from_fp16, 0, sizeof(cl_mem), &in_d);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel_convert_from_fp16, 1, out);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convert_from_fp16, 2, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convert_from_fp16, 3, sizeof(int), &W);
  CHECK_ERROR(err);
  
  size_t gws_1d, lws_1d;
  gws_1d = H * W;
  lws_1d = 384;
  gws_1d = (gws_1d + lws_1d - 1) / lws_1d * lws_1d;

  err = clEnqueueNDRangeKernel(queue, kernel_convert_from_fp16, 1, NULL,
      &gws_1d, &lws_1d, 0, NULL, &event);
  CHECK_ERROR(err);
#ifdef kernel_time
  print_kernel_time("convert_from_fp16()");
#endif
}
#endif

void read_gpu_buffer() {
  timer_start(0);

  clEnqueueSVMUnmap(queue, output_green, 0, NULL, &event);
  clEnqueueSVMUnmap(queue, output_red, 0, NULL, &event);
  clEnqueueSVMUnmap(queue, output_blue, 0, NULL, &event);

  clFinish(queue);
  printf("read_gpu_buffer() complete : %lf sec\n", timer_stop(0));
}

void demosaic()  {
  timer_start(0);

  green_interpolation();
  red_blue_interpolation();

  clFinish(queue);
  printf("demosaic time : %lf sec\n", timer_stop(0));
}

void green_interpolation() {
#ifdef USE_HALF_PRECISON
  convert_to_fp16(input_rgb, raw_d);
#else
  raw_d = input_rgb;
#endif

  imfilter2(raw_d, raw_H_d, raw_V_d);

  linear_coefficient_horizontal(raw_d, raw_H_d, 
      A_H_d, B_H_d, C_H_d, D_H_d);
  linear_coefficient_vertical(raw_d, raw_V_d, 
      A_V_d, B_V_d, C_V_d, D_V_d);

  tentative_residual_horizontal(raw_d, raw_H_d, A_H_d, B_H_d, C_H_d, D_H_d, 
      tentative_H_d, residual_H_d);
  tentative_residual_vertical(raw_d, raw_V_d, A_V_d, B_V_d, C_V_d, D_V_d, 
      tentative_V_d, residual_V_d);

  residual_interpolation_and_add_tentative(raw_d, residual_H_d, residual_V_d, 
      tentative_H_d, tentative_V_d);

  color_difference_h(tentative_H_d, A_H_d, C_V_d, D_V_d);
  color_difference_v(tentative_V_d, B_H_d, A_V_d, B_V_d);

  compute_weight_new(A_H_d, C_H_d);
  compute_weight_new(B_H_d, D_H_d);

  directional_weight(raw_d, A_V_d, B_V_d, C_V_d, D_V_d, C_H_d, D_H_d, raw_H_d);
}

void red_blue_interpolation() {
  red_blue_vertical_sum(raw_H_d, raw_d, 
      A_V_d, B_V_d, C_V_d, D_V_d);

  horizontal_sum_and_linear_coefficient(A_V_d, B_V_d, C_V_d, D_V_d,
      A_H_d, B_H_d, C_H_d, D_H_d);

  coefficient_vertical_sum(A_H_d, A_V_d);
  coefficient_vertical_sum(B_H_d, B_V_d);
  coefficient_vertical_sum(C_H_d, C_V_d);
  coefficient_vertical_sum(D_H_d, D_V_d);

  horizontal_sum_and_tentative_residual(raw_H_d, raw_d, A_V_d, B_V_d, C_V_d, D_V_d, 
      tentative_H_d, tentative_V_d, residual_H_d);

  red_blue_residual_interpolation_and_add_tentative(residual_H_d,
      tentative_H_d, tentative_V_d);

#ifdef USE_HALF_PRECISON
  convert_from_fp16(raw_H_d, output_green);
  convert_from_fp16(tentative_H_d, output_red);
  convert_from_fp16(tentative_V_d, output_blue);
#else
  output_green = raw_H_d;
  output_red = tentative_H_d;
  output_blue = tentative_V_d;
#endif
}

void free_resources() {
  timer_start(0);

  clSVMFree(queue, input_rgb);

#ifdef USE_HALF_PRECISON
  clSVMFree(queue, output_green);
  clSVMFree(queue, output_red);
  clSVMFree(queue, output_blue);

  clReleaseMemObject(raw_H_d);
  clReleaseMemObject(tentative_H_d);
  clReleaseMemObject(tentative_V_d);
#else
  clSVMFree(queue, raw_H_d);
  clSVMFree(queue, tentative_H_d);
  clSVMFree(queue, tentative_V_d);
#endif

  clReleaseMemObject(raw_V_d);
  clReleaseMemObject(A_H_d);
  clReleaseMemObject(B_H_d);
  clReleaseMemObject(C_H_d);
  clReleaseMemObject(D_H_d);
  clReleaseMemObject(A_V_d);
  clReleaseMemObject(B_V_d);
  clReleaseMemObject(C_V_d);
  clReleaseMemObject(D_V_d);
  clReleaseMemObject(residual_H_d);
  clReleaseMemObject(residual_V_d);

  clReleaseKernel(kernel_imfilter2);
  clReleaseKernel(kernel_residual_interpolation);
  clReleaseKernel(kernel_compute_weight_new);
  clReleaseKernel(kernel_directional_weight);
  clReleaseKernel(kernel_linear_coefficient_h);
  clReleaseKernel(kernel_linear_coefficient_v);
  clReleaseKernel(kernel_tentative_residual_h);
  clReleaseKernel(kernel_tentative_residual_v);
  clReleaseKernel(kernel_color_difference_h);
  clReleaseKernel(kernel_color_difference_v);
  clReleaseKernel(kernel_red_blue_vertical_sum);
  clReleaseKernel(kernel_horizontal_sum_and_linear_coefficient);
  clReleaseKernel(kernel_coefficient_vertical_sum);
  clReleaseKernel(kernel_horizontal_sum_and_tentative_residual);
  clReleaseKernel(kernel_red_blue_residual_interpolation_and_add_tentative);

#ifdef USE_HALF_PRECISON
  clReleaseKernel(kernel_convert_to_fp16);
  clReleaseKernel(kernel_convert_from_fp16);
#endif

  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  printf("free resources complete : %lf\n", timer_stop(0));
}

void* read_file(const char *fn, size_t *sz) {
  size_t sz_;
  FILE *f = fopen(fn, "rb");
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  sz_ = ftell(f);
  rewind(f);

  void *buf = malloc(sz_);
  size_t ret = fread(buf, 1, sz_, f);
  fclose(f);

  if (sz_ != ret) return NULL;
  if (sz != NULL) *sz = sz_;
  return buf;
}

int diffrgb_float(const char*fn1, const char* fn2) {
  size_t sz1, sz2;
  float *buf1 = (float *)read_file(fn1, &sz1);
  float *buf2 = (float *)read_file(fn2, &sz2);

  printf("file1 : %s, file2 : %s\n", fn1, fn2);
  if (sz1 != sz2) {
    printf("Size should be the same. (%zu != %zu)\n", sz1, sz2);
    exit(0);
  }
  float mse = 0;

  for (size_t i = 0; i < H*W*C; ++i) {
      float diff = buf1[i] - buf2[i];

      if (isnan(buf1[i]) || isnan(buf2[i])) {
        printf("ERROR: isnan found!!!!!!!!!!\n");
        return -1;
      }

	  mse += diff*diff;
  }

  mse = mse / sz1;
  printf("Out of %zu pixels...\n", sz1);
  printf("mean square error : %f\n", mse);

  return 0;
}

int diffrgb(const char*fn1, const char* fn2) {
  size_t sz1, sz2;
  unsigned char *buf1 = (unsigned char *)read_file(fn1, &sz1);
  unsigned char *buf2 = (unsigned char*)read_file(fn2, &sz2);

  printf("file1 : %s, file2 : %s\n", fn1, fn2);
  if (sz1 != sz2) {
    printf("Size should be the same. (%zu != %zu)\n", sz1, sz2);
    exit(0);
  }
  int mse = 0;

  for (size_t i = 0; i < H*W*C; ++i) {
      int diff = buf1[i] - buf2[i];
	  mse += diff*diff;
  }

  mse = mse / sz1;
  printf("Out of %zu pixels...\n", sz1);
  printf("mean square error : %d\n", mse);

  return 0;
}

void validation() {

  // RGB
  //write_file("image/my.buf");
  //diffrgb("image/my.buf", "image/target.buf");

  // float
  write_file_float("image/my_float.buf");
  diffrgb_float("image/my_float.buf", "image/target_float2.buf");
}
