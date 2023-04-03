#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void convert_to_fp16(__global float *in, __global half *out, int H, int W) {
  int gid = get_global_id(0);
  if (gid >= H*W) return;

  float val = in[gid];
  vstore_half(val, gid, out);
}

__kernel void convert_from_fp16(__global half *in, __global float *out, int H, int W) {
  int gid = get_global_id(0);
  if (gid >= H*W) return;

  float val = vload_half(gid, in);
  out[gid] = val;
}
