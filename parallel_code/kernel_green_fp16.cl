#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define THRESHOLD  0.65025
#define WT 0.570350
#define WT1 0.345934
#define WT2 0.077188
#define WT3 0.006336
#define WT4 0.000192


__kernel void imfilter2(__global half *in, __global half* out1, __global half* out2,
                        int H, int W) {
  int i = get_global_id(0);
  if( i >= H*W ) return;

  int oh = i/W;
  int ow = i%W;

  half v = in[i];
  half x;

  int ii = ow-1;
  if ( ii < 0 ) x = v;
  else x = in[i-1];

  ii = ow+1;
  if ( ii >= W ) x += v;
  else x += in[i+1];

  out1[i] = x / 2;

  // for vertical
  ii = oh-1;
  if ( ii < 0 ) x = v;
  else x = in[ii*W+ow];

  ii = oh+1;
  if ( ii >= H ) x += v;
  else x += in[ii*W+ow];

  out2[i] = x / 2;

}

/*
// half8
__kernel void linear_coefficient_horizontal(__global half4 *P, __global half4 *I,
                                            __global half *A_H, __global half *B_H,
                                            __global half *C_H, __global half *D_H,
                                            int H, int W) {
  int i = get_global_id(0);
  if (i >= H * W) return;
  int oh = i / W;
  int ow = i - oh*W;

  int quiotient = i >> 3;
  int remainder = i & 0x1;
  // vectors: I_pre2, I_pre1, I_mid, I_post1, I_post2
  // vectors: P_pre2, P_pre1, P_mid, P_post1, P_post2

  half I_sum5, I_sum6, P_sum5, P_sum6;
  I_sum5 = I_sum6 = P_sum5 = P_sum6 = 0;

  half II_sum5, II_sum6, IP_sum5, IP_sum6;
  II_sum5 = II_sum6 = IP_sum5 = IP_sum6 = 0;

  unsigned int divisor5, divisor6;
  divisor5 = divisor6 = 0;

  /////////////////////////////////////////// pre2 //
  if (ow >= 8) {
    half4 I_pre2, P_pre2;

    I_pre2 = I[quiotient - 2];
    P_pre2 = P[quiotient - 2];

    if (remainder == 0) {
      I_sum6 += I_pre2.w;
      P_sum6 += P_pre2.w;
      II_sum6 += I_pre2.w*I_pre2.w;
      IP_sum6 += I_pre2.w*P_pre2.w;
	  divisor6 += 1;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    half4 I_pre1, P_pre1;

    I_pre1 = I[quiotient - 1];
    P_pre1 = P[quiotient - 1];

    if (remainder == 0) {
      I_sum5 += I_pre1.x;
      I_sum6 += I_pre1.y;
      P_sum5 += P_pre1.x;
      P_sum6 += P_pre1.y;

      II_sum5 += I_pre1.x * I_pre1.x;
      II_sum6 += I_pre1.y * I_pre1.y;
      IP_sum5 += I_pre1.x * P_pre1.x;
      IP_sum6 += I_pre1.y * P_pre1.y;
    }
    else {
      I_sum5 += I_pre1.y;
      I_sum6 += I_pre1.x;
      P_sum5 += P_pre1.y;
      P_sum6 += P_pre1.x;

      II_sum5 += I_pre1.y * I_pre1.y;
      II_sum6 += I_pre1.x * I_pre1.x;
      IP_sum5 += I_pre1.y * P_pre1.y;
      IP_sum6 += I_pre1.x * P_pre1.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  /////////////////////////////////////////// mid //
  {
    half4 I_mid = I[quiotient];
    half4 P_mid = P[quiotient];

    if (remainder == 0) {
      I_sum5 += I_mid.x;
      I_sum6 += I_mid.y;
      P_sum5 += P_mid.x;
      P_sum6 += P_mid.y;

      II_sum5 += I_mid.x * I_mid.x;
      II_sum6 += I_mid.y * I_mid.y;
      IP_sum5 += I_mid.x * P_mid.x;
      IP_sum6 += I_mid.y * P_mid.y;
    }
    else {
      I_sum5 += I_mid.y;
      I_sum6 += I_mid.x;
      P_sum5 += P_mid.y;
      P_sum6 += P_mid.x;

      II_sum5 += I_mid.y * I_mid.y;
      II_sum6 += I_mid.x * I_mid.x;
      IP_sum5 += I_mid.y * P_mid.y;
      IP_sum6 += I_mid.x * P_mid.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    half4 I_post1, P_post1;

    I_post1 = I[quiotient + 1];
    P_post1 = P[quiotient + 1];

    if (remainder == 0) {
      I_sum5 += I_post1.x;
      I_sum6 += I_post1.y;
      P_sum5 += P_post1.x;
      P_sum6 += P_post1.y;

      II_sum5 += I_post1.x * I_post1.x;
      II_sum6 += I_post1.y * I_post1.y;
      IP_sum5 += I_post1.x * P_post1.x;
      IP_sum6 += I_post1.y * P_post1.y;
    }
    else {
      I_sum5 += I_post1.y;
      I_sum6 += I_post1.x;
      P_sum5 += P_post1.y;
      P_sum6 += P_post1.x;

      II_sum5 += I_post1.y * I_post1.y;
      II_sum6 += I_post1.x * I_post1.x;
      IP_sum5 += I_post1.y * P_post1.y;
      IP_sum6 += I_post1.x * P_post1.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  ////////////////////////////////////////// post2 //
  if (ow < W-8) {
    half4 I_post2, P_post2;

    I_post2 = I[quiotient + 2];
    P_post2 = P[quiotient + 2];

    if (remainder == 0) {
      I_sum6 += I_post2.x;
      P_sum6 += P_post2.x;
      II_sum6 += I_post2.x*I_post2.x;
      IP_sum6 += I_post2.x*P_post2.x;
    }
    divisor6 += 1;
  }
  ////////////////////////////////////////////////////
  I_sum5 = I_sum5 / divisor5; 
  P_sum5 = P_sum5 / divisor5; 
  II_sum5 = II_sum5 / divisor5; 
  IP_sum5 = IP_sum5 / divisor5; 

  half varianceI, covarianceIP, coefficient;
  covarianceIP = IP_sum5 - I_sum5 * P_sum5;
  varianceI = II_sum5 - I_sum5 * I_sum5;
  varianceI = (varianceI < THRESHOLD) ? THRESHOLD : varianceI;
  coefficient = covarianceIP / varianceI;
  A_H[i] = coefficient;
  B_H[i] = P_sum5 - coefficient * I_sum5;

  I_sum6 = I_sum6 / divisor6;
  P_sum6 = P_sum6 / divisor6;
  II_sum6 = II_sum6 / divisor6;
  IP_sum6 = IP_sum6 / divisor6;

  covarianceIP = IP_sum6 - I_sum6 * P_sum6;
  varianceI = II_sum6 - I_sum6 * I_sum6;
  varianceI = (varianceI < THRESHOLD) ? THRESHOLD : varianceI;
  coefficient = covarianceIP / varianceI;
  C_H[i] = coefficient;
  D_H[i] = P_sum6 - coefficient * I_sum6;
}
*/


// half4
__kernel void _linear_coefficient_horizontal(__global half4 *P, __global half4 *I,
                                            __global half *A_H, __global half *B_H,
                                            __global half *C_H, __global half *D_H,
                                            int H, int W) {
  int i = get_global_id(0);
  if (i >= H * W) return;
  int oh = i / W;
  int ow = i - oh*W;

  int quiotient = i >> 2;
  int remainder = i & 0x1;
  // vectors: I_pre2, I_pre1, I_mid, I_post1, I_post2
  // vectors: P_pre2, P_pre1, P_mid, P_post1, P_post2

  half I_sum5, I_sum6, P_sum5, P_sum6;
  I_sum5 = I_sum6 = P_sum5 = P_sum6 = 0;

  half II_sum5, II_sum6, IP_sum5, IP_sum6;
  II_sum5 = II_sum6 = IP_sum5 = IP_sum6 = 0;

  unsigned int divisor5, divisor6;
  divisor5 = divisor6 = 0;

  /////////////////////////////////////////// pre2 //
  if (ow >= 8) {
    half4 I_pre2, P_pre2;

    I_pre2 = I[quiotient - 2];
    P_pre2 = P[quiotient - 2];

    if (remainder == 0) {
      I_sum6 += I_pre2.w;
      P_sum6 += P_pre2.w;
      II_sum6 += I_pre2.w*I_pre2.w;
      IP_sum6 += I_pre2.w*P_pre2.w;
	  divisor6 += 1;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    half4 I_pre1, P_pre1;

    I_pre1 = I[quiotient - 1];
    P_pre1 = P[quiotient - 1];

    if (remainder == 0) {
      I_sum5 += I_pre1.x;
      I_sum6 += I_pre1.y;
      P_sum5 += P_pre1.x;
      P_sum6 += P_pre1.y;

      II_sum5 += I_pre1.x * I_pre1.x;
      II_sum6 += I_pre1.y * I_pre1.y;
      IP_sum5 += I_pre1.x * P_pre1.x;
      IP_sum6 += I_pre1.y * P_pre1.y;
    }
    else {
      I_sum5 += I_pre1.y;
      I_sum6 += I_pre1.x;
      P_sum5 += P_pre1.y;
      P_sum6 += P_pre1.x;

      II_sum5 += I_pre1.y * I_pre1.y;
      II_sum6 += I_pre1.x * I_pre1.x;
      IP_sum5 += I_pre1.y * P_pre1.y;
      IP_sum6 += I_pre1.x * P_pre1.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  /////////////////////////////////////////// mid //
  {
    half4 I_mid = I[quiotient];
    half4 P_mid = P[quiotient];

    if (remainder == 0) {
      I_sum5 += I_mid.x;
      I_sum6 += I_mid.y;
      P_sum5 += P_mid.x;
      P_sum6 += P_mid.y;

      II_sum5 += I_mid.x * I_mid.x;
      II_sum6 += I_mid.y * I_mid.y;
      IP_sum5 += I_mid.x * P_mid.x;
      IP_sum6 += I_mid.y * P_mid.y;
    }
    else {
      I_sum5 += I_mid.y;
      I_sum6 += I_mid.x;
      P_sum5 += P_mid.y;
      P_sum6 += P_mid.x;

      II_sum5 += I_mid.y * I_mid.y;
      II_sum6 += I_mid.x * I_mid.x;
      IP_sum5 += I_mid.y * P_mid.y;
      IP_sum6 += I_mid.x * P_mid.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    half4 I_post1, P_post1;

    I_post1 = I[quiotient + 1];
    P_post1 = P[quiotient + 1];

    if (remainder == 0) {
      I_sum5 += I_post1.x;
      I_sum6 += I_post1.y;
      P_sum5 += P_post1.x;
      P_sum6 += P_post1.y;

      II_sum5 += I_post1.x * I_post1.x;
      II_sum6 += I_post1.y * I_post1.y;
      IP_sum5 += I_post1.x * P_post1.x;
      IP_sum6 += I_post1.y * P_post1.y;
    }
    else {
      I_sum5 += I_post1.y;
      I_sum6 += I_post1.x;
      P_sum5 += P_post1.y;
      P_sum6 += P_post1.x;

      II_sum5 += I_post1.y * I_post1.y;
      II_sum6 += I_post1.x * I_post1.x;
      IP_sum5 += I_post1.y * P_post1.y;
      IP_sum6 += I_post1.x * P_post1.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  ////////////////////////////////////////// post2 //
  if (ow < W-8) {
    half4 I_post2, P_post2;

    I_post2 = I[quiotient + 2];
    P_post2 = P[quiotient + 2];

    if (remainder == 0) {
      I_sum6 += I_post2.x;
      P_sum6 += P_post2.x;
      II_sum6 += I_post2.x*I_post2.x;
      IP_sum6 += I_post2.x*P_post2.x;
    }
    divisor6 += 1;
  }
  ////////////////////////////////////////////////////
  I_sum5 = I_sum5 / divisor5; 
  P_sum5 = P_sum5 / divisor5; 
  II_sum5 = II_sum5 / divisor5; 
  IP_sum5 = IP_sum5 / divisor5; 

  half varianceI, covarianceIP, coefficient;
  covarianceIP = IP_sum5 - I_sum5 * P_sum5;
  varianceI = II_sum5 - I_sum5 * I_sum5;
  varianceI = (varianceI < THRESHOLD) ? THRESHOLD : varianceI;
  coefficient = covarianceIP / varianceI;
  A_H[i] = coefficient;
  B_H[i] = P_sum5 - coefficient * I_sum5;

  I_sum6 = I_sum6 / divisor6;
  P_sum6 = P_sum6 / divisor6;
  II_sum6 = II_sum6 / divisor6;
  IP_sum6 = IP_sum6 / divisor6;

  covarianceIP = IP_sum6 - I_sum6 * P_sum6;
  varianceI = II_sum6 - I_sum6 * I_sum6;
  varianceI = (varianceI < THRESHOLD) ? THRESHOLD : varianceI;
  coefficient = covarianceIP / varianceI;
  C_H[i] = coefficient;
  D_H[i] = P_sum6 - coefficient * I_sum6;
}


// half2
__kernel void linear_coefficient_horizontal(__global half2 *P, __global half2 *I,
                                            __global half *A_H, __global half *B_H,
                                            __global half *C_H, __global half *D_H,
                                            int H, int W) {
  int i = get_global_id(0);
  if (i >= H * W) return;
  int oh = i / W;
  int ow = i - oh*W;

  int quiotient = i >> 1;
  int remainder = i & 0x1;
  // vectors: I_pre3, I_pre2, I_pre1, I_mid, I_post1, I_post2, I_post3
  // vectors: P_pre3, P_pre2, P_pre1, P_mid, P_post1, P_post2, P_post3

  half I_sum5, I_sum6, P_sum5, P_sum6;
  I_sum5 = I_sum6 = P_sum5 = P_sum6 = 0;

  half II_sum5, II_sum6, IP_sum5, IP_sum6;
  II_sum5 = II_sum6 = IP_sum5 = IP_sum6 = 0;

  unsigned int divisor5, divisor6;
  divisor5 = divisor6 = 0;

  /////////////////////////////////////////// pre3 //
  if (ow >= 6) {
    if (remainder == 0) {
      half I_v, P_v;
      I_v = I[quiotient - 3].y;
      P_v = P[quiotient - 3].y;
      I_sum6 += I_v;
      P_sum6 += P_v;
      II_sum6 += I_v * I_v;
      IP_sum6 += I_v * P_v;
      divisor6 += 1; 
    }
  }

  /////////////////////////////////////////// pre2 //
  if (ow >= 4) {
    half2 I_pre2, P_pre2;

    I_pre2 = I[quiotient - 2];
    P_pre2 = P[quiotient - 2];

    if (remainder == 0) {
      I_sum5 += I_pre2.x;
      I_sum6 += I_pre2.y;
      P_sum5 += P_pre2.x;
      P_sum6 += P_pre2.y;

      II_sum5 += I_pre2.x * I_pre2.x;
      II_sum6 += I_pre2.y * I_pre2.y;
      IP_sum5 += I_pre2.x * P_pre2.x;
      IP_sum6 += I_pre2.y * P_pre2.y;
    }
    else {
      I_sum5 += I_pre2.y;
      I_sum6 += I_pre2.x;
      P_sum5 += P_pre2.y;
      P_sum6 += P_pre2.x;

      II_sum5 += I_pre2.y * I_pre2.y;
      II_sum6 += I_pre2.x * I_pre2.x;
      IP_sum5 += I_pre2.y * P_pre2.y;
      IP_sum6 += I_pre2.x * P_pre2.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 2) {
    half2 I_pre1, P_pre1;

    I_pre1 = I[quiotient - 1];
    P_pre1 = P[quiotient - 1];

    if (remainder == 0) {
      I_sum5 += I_pre1.x;
      I_sum6 += I_pre1.y;
      P_sum5 += P_pre1.x;
      P_sum6 += P_pre1.y;

      II_sum5 += I_pre1.x * I_pre1.x;
      II_sum6 += I_pre1.y * I_pre1.y;
      IP_sum5 += I_pre1.x * P_pre1.x;
      IP_sum6 += I_pre1.y * P_pre1.y;
    }
    else {
      I_sum5 += I_pre1.y;
      I_sum6 += I_pre1.x;
      P_sum5 += P_pre1.y;
      P_sum6 += P_pre1.x;

      II_sum5 += I_pre1.y * I_pre1.y;
      II_sum6 += I_pre1.x * I_pre1.x;
      IP_sum5 += I_pre1.y * P_pre1.y;
      IP_sum6 += I_pre1.x * P_pre1.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  /////////////////////////////////////////// mid //
  {
    half2 I_mid = I[quiotient];
    half2 P_mid = P[quiotient];

    if (remainder == 0) {
      I_sum5 += I_mid.x;
      I_sum6 += I_mid.y;
      P_sum5 += P_mid.x;
      P_sum6 += P_mid.y;

      II_sum5 += I_mid.x * I_mid.x;
      II_sum6 += I_mid.y * I_mid.y;
      IP_sum5 += I_mid.x * P_mid.x;
      IP_sum6 += I_mid.y * P_mid.y;
    }
    else {
      I_sum5 += I_mid.y;
      I_sum6 += I_mid.x;
      P_sum5 += P_mid.y;
      P_sum6 += P_mid.x;

      II_sum5 += I_mid.y * I_mid.y;
      II_sum6 += I_mid.x * I_mid.x;
      IP_sum5 += I_mid.y * P_mid.y;
      IP_sum6 += I_mid.x * P_mid.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-2) {
    half2 I_post1, P_post1;

    I_post1 = I[quiotient + 1];
    P_post1 = P[quiotient + 1];

    if (remainder == 0) {
      I_sum5 += I_post1.x;
      I_sum6 += I_post1.y;
      P_sum5 += P_post1.x;
      P_sum6 += P_post1.y;

      II_sum5 += I_post1.x * I_post1.x;
      II_sum6 += I_post1.y * I_post1.y;
      IP_sum5 += I_post1.x * P_post1.x;
      IP_sum6 += I_post1.y * P_post1.y;
    }
    else {
      I_sum5 += I_post1.y;
      I_sum6 += I_post1.x;
      P_sum5 += P_post1.y;
      P_sum6 += P_post1.x;

      II_sum5 += I_post1.y * I_post1.y;
      II_sum6 += I_post1.x * I_post1.x;
      IP_sum5 += I_post1.y * P_post1.y;
      IP_sum6 += I_post1.x * P_post1.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  ////////////////////////////////////////// post2 //
  if (ow < W-4) {
    half2 I_post2, P_post2;

    I_post2 = I[quiotient + 2];
    P_post2 = P[quiotient + 2];

    if (remainder == 0) {
      I_sum5 += I_post2.x;
      I_sum6 += I_post2.y;
      P_sum5 += P_post2.x;
      P_sum6 += P_post2.y;

      II_sum5 += I_post2.x * I_post2.x;
      II_sum6 += I_post2.y * I_post2.y;
      IP_sum5 += I_post2.x * P_post2.x;
      IP_sum6 += I_post2.y * P_post2.y;
    }
    else {
      I_sum5 += I_post2.y;
      I_sum6 += I_post2.x;
      P_sum5 += P_post2.y;
      P_sum6 += P_post2.x;

      II_sum5 += I_post2.y * I_post2.y;
      II_sum6 += I_post2.x * I_post2.x;
      IP_sum5 += I_post2.y * P_post2.y;
      IP_sum6 += I_post2.x * P_post2.x;
    }
    divisor5 += 1;
    divisor6 += 1;
  }

  ////////////////////////////////////////// post3 //
  if (ow < W-6) {
    if (remainder == 1) {
      half I_v, P_v;
      I_v = I[quiotient + 3].x;
      P_v = P[quiotient + 3].x;

      I_sum6 += I_v;
      P_sum6 += P_v;
      II_sum6 += I_v * I_v;
      IP_sum6 += I_v * P_v;
      divisor6 += 1;
    }
  }

  ////////////////////////////////////////////////////
  I_sum5 = I_sum5 / divisor5; 
  P_sum5 = P_sum5 / divisor5; 
  II_sum5 = II_sum5 / divisor5; 
  IP_sum5 = IP_sum5 / divisor5; 

  half varianceI, covarianceIP, coefficient;
  covarianceIP = IP_sum5 - I_sum5 * P_sum5;
  varianceI = II_sum5 - I_sum5 * I_sum5;
  varianceI = (varianceI < THRESHOLD) ? THRESHOLD : varianceI;
  coefficient = covarianceIP / varianceI;
  A_H[i] = coefficient;
  B_H[i] = P_sum5 - coefficient * I_sum5;

  I_sum6 = I_sum6 / divisor6;
  P_sum6 = P_sum6 / divisor6;
  II_sum6 = II_sum6 / divisor6;
  IP_sum6 = IP_sum6 / divisor6;

  covarianceIP = IP_sum6 - I_sum6 * P_sum6;
  varianceI = II_sum6 - I_sum6 * I_sum6;
  varianceI = (varianceI < THRESHOLD) ? THRESHOLD : varianceI;
  coefficient = covarianceIP / varianceI;
  C_H[i] = coefficient;
  D_H[i] = P_sum6 - coefficient * I_sum6;
}


__kernel void linear_coefficient_vertical(__global half2 *P, __global half2 *I,
                                          __global half2 *A, __global half2 *B,
                                          __global half2 *C, __global half2 *D,
                                          int H, int W) {
    int i = get_global_id(0);
    if (i*2 >= W) return; // boundary check 

    half2 I_v5_=0, I_v4_=0, I_v3_=0, I_v2_=0, I_v1_=0;
    half2 I_v, I_v1, I_v2, I_v3, I_v4, I_v5;
    half2 P_v5_=0, P_v4_=0, P_v3_=0, P_v2_=0, P_v1_=0;
    half2 P_v, P_v1, P_v2, P_v3, P_v4, P_v5, P_v6;
    half2 I_sum5, I_sum6, P_sum5, P_sum6, II_sum5, II_sum6, IP_sum5, IP_sum6;

    int _W = W/2;

    I_v5_ = I_v4_ = I_v3_ = I_v2_ = I_v1_ = (half2)0;
    P_v5_ = P_v4_ = P_v3_ = P_v2_ = P_v1_ = (half2)0;

    ///////////// first sum
    /////////////////////////////////////////// first sum
    int idx = i;
    P_v = P[idx];

    idx += _W;
    P_v1 = P[idx];

    idx += _W;
    P_v2 = P[idx];

    idx += _W;
    P_v3 = P[idx];

    idx += _W;
    P_v4 = P[idx];

    idx += _W;
    P_v5 = P[idx];

    // I
    idx = i;
    I_v = I[idx];

    idx += _W;
    I_v1 = I[idx];

    idx += _W;
    I_v2 = I[idx];

    idx += _W;
    I_v3 = I[idx];

    idx += _W;
    I_v4 = I[idx];

    idx += _W;
    I_v5 = I[idx];
    /////

    I_sum5 = (I_v + I_v2 + I_v4) / 3;
    I_sum6 = (I_v1 + I_v3 + I_v5) / 3;
    P_sum5 = (P_v + P_v2 + P_v4) / 3;  
    P_sum6 = (P_v1 + P_v3 + P_v5) / 3;

    II_sum5 = (I_v*I_v + I_v2*I_v2 + I_v4*I_v4) / 3;
    II_sum6 = (I_v1*I_v1 + I_v3*I_v3 + I_v5*I_v5) / 3;
    IP_sum5 = (I_v*P_v + I_v2*P_v2 + I_v4*P_v4) / 3;
    IP_sum6 = (I_v1*P_v1 + I_v3*P_v3 + I_v5*P_v5) / 3;

    half2 varianceI, covarianceIP, coefficient;

    covarianceIP = IP_sum5 - I_sum5 * P_sum5;
    varianceI = II_sum5 - I_sum5 * I_sum5;
    varianceI.x = (varianceI.x < THRESHOLD) ? THRESHOLD : varianceI.x;
    varianceI.y = (varianceI.y < THRESHOLD) ? THRESHOLD : varianceI.y;
    coefficient = covarianceIP / varianceI;
    A[i] = coefficient;
    B[i] = P_sum5 - coefficient * I_sum5;

    covarianceIP = IP_sum6 - I_sum6 * P_sum6;
    varianceI = II_sum6 - I_sum6 * I_sum6;
    varianceI.x = (varianceI.x < THRESHOLD) ? THRESHOLD : varianceI.x;
    varianceI.y = (varianceI.y < THRESHOLD) ? THRESHOLD : varianceI.y;
    coefficient = covarianceIP / varianceI;
    C[i] = coefficient;
    D[i] = P_sum6 - coefficient * I_sum6;

    //////////////////////////////////////////////////// next sum
    int divisor5 = 3, divisor6 = 3;

    for( int oh = 1; oh < 5; oh ++ ) {
        i += _W;

        if ((oh & 0x1) == 1)
          divisor6 += 1;
        else
          divisor5 += 1;

        // pull register (v5_~v5)
        I_v5_ = I_v4_;
        I_v4_ = I_v3_;
        I_v3_ = I_v2_;
        I_v2_ = I_v1_;
        I_v1_ = I_v;
        I_v = I_v1;
        I_v1 = I_v2;
        I_v2 = I_v3;
        I_v3 = I_v4;
        I_v4 = I_v5;  

        P_v5_ = P_v4_;
        P_v4_ = P_v3_;
        P_v3_ = P_v2_;
        P_v2_ = P_v1_;
        P_v1_ = P_v;
        P_v = P_v1;
        P_v1 = P_v2;
        P_v2 = P_v3;
        P_v3 = P_v4;
        P_v4 = P_v5;  

        ///////////////////////// new
        idx = i + 5*_W;

        I_v5 = I[idx];
        P_v5 = P[idx];

        I_sum5 = (I_v4_ + I_v2_ + I_v + I_v2 + I_v4) / divisor5;
        I_sum6 = (I_v5_ + I_v3_ + I_v1_ + I_v1 + I_v3 + I_v5) / divisor6;
        P_sum5 = (P_v4_ + P_v2_ + P_v + P_v2 + P_v4) / divisor5;
        P_sum6 = (P_v5_ + P_v3_ + P_v1_ + P_v1 + P_v3 + P_v5) / divisor6;

        II_sum5 = (I_v4_*I_v4_ + I_v2_*I_v2_ + I_v*I_v + I_v2*I_v2 + I_v4*I_v4) / divisor5;
        II_sum6 = (I_v5_*I_v5_ + I_v3_*I_v3_ + I_v1_*I_v1_ + I_v1*I_v1 + I_v3*I_v3 + I_v5*I_v5) / divisor6;
        IP_sum5 = (I_v4_*P_v4_ + I_v2_*P_v2_ + I_v*P_v + I_v2*P_v2 + I_v4*P_v4) / divisor5;
        IP_sum6 = (I_v5_*P_v5_ + I_v3_*P_v3_ + I_v1_*P_v1_ + I_v1*P_v1 + I_v3*P_v3 + I_v5*P_v5) / divisor6;
      
        covarianceIP = IP_sum5 - I_sum5 * P_sum5;
        varianceI = II_sum5 - I_sum5 * I_sum5;
        varianceI.x = (varianceI.x < THRESHOLD) ? THRESHOLD : varianceI.x;
        varianceI.y = (varianceI.y < THRESHOLD) ? THRESHOLD : varianceI.y;
        coefficient = covarianceIP / varianceI;
        A[i] = coefficient;
        B[i] = P_sum5 - coefficient * I_sum5;

        covarianceIP = IP_sum6 - I_sum6 * P_sum6;
        varianceI = II_sum6 - I_sum6 * I_sum6;
        varianceI.x = (varianceI.x < THRESHOLD) ? THRESHOLD : varianceI.x;
        varianceI.y = (varianceI.y < THRESHOLD) ? THRESHOLD : varianceI.y;
        coefficient = covarianceIP / varianceI;
        C[i] = coefficient;
        D[i] = P_sum6 - coefficient * I_sum6;
    }

    for( int oh = 5; oh < H-5; oh ++ ) {
        i += _W;

        // pull register (v5_~v5)
        I_v5_ = I_v4_;
        I_v4_ = I_v3_;
        I_v3_ = I_v2_;
        I_v2_ = I_v1_;
        I_v1_ = I_v;
        I_v = I_v1;
        I_v1 = I_v2;
        I_v2 = I_v3;
        I_v3 = I_v4;
        I_v4 = I_v5;  

        P_v5_ = P_v4_;
        P_v4_ = P_v3_;
        P_v3_ = P_v2_;
        P_v2_ = P_v1_;
        P_v1_ = P_v;
        P_v = P_v1;
        P_v1 = P_v2;
        P_v2 = P_v3;
        P_v3 = P_v4;
        P_v4 = P_v5;  

        ///////////////////////// new
        idx = i + 5*_W;

        I_v5 = I[idx];
        P_v5 = P[idx];

        I_sum5 = (I_v4_ + I_v2_ + I_v + I_v2 + I_v4) / 5;
        I_sum6 = (I_v5_ + I_v3_ + I_v1_ + I_v1 + I_v3 + I_v5) / 6;
        P_sum5 = (P_v4_ + P_v2_ + P_v + P_v2 + P_v4) / 5;
        P_sum6 = (P_v5_ + P_v3_ + P_v1_ + P_v1 + P_v3 + P_v5) / 6;

        II_sum5 = (I_v4_*I_v4_ + I_v2_*I_v2_ + I_v*I_v + I_v2*I_v2 + I_v4*I_v4) / 5;
        II_sum6 = (I_v5_*I_v5_ + I_v3_*I_v3_ + I_v1_*I_v1_ + I_v1*I_v1 + I_v3*I_v3 + I_v5*I_v5) / 6;
        IP_sum5 = (I_v4_*P_v4_ + I_v2_*P_v2_ + I_v*P_v + I_v2*P_v2 + I_v4*P_v4) / 5;
        IP_sum6 = (I_v5_*P_v5_ + I_v3_*P_v3_ + I_v1_*P_v1_ + I_v1*P_v1 + I_v3*P_v3 + I_v5*P_v5) / 6;

        covarianceIP = IP_sum5 - I_sum5 * P_sum5;
        varianceI = II_sum5 - I_sum5 * I_sum5;
        varianceI.x = (varianceI.x < THRESHOLD) ? THRESHOLD : varianceI.x;
        varianceI.y = (varianceI.y < THRESHOLD) ? THRESHOLD : varianceI.y;
        coefficient = covarianceIP / varianceI;
        A[i] = coefficient;
        B[i] = P_sum5 - coefficient * I_sum5;

        covarianceIP = IP_sum6 - I_sum6 * P_sum6;
        varianceI = II_sum6 - I_sum6 * I_sum6;
        varianceI.x = (varianceI.x < THRESHOLD) ? THRESHOLD : varianceI.x;
        varianceI.y = (varianceI.y < THRESHOLD) ? THRESHOLD : varianceI.y;
        coefficient = covarianceIP / varianceI;
        C[i] = coefficient;
        D[i] = P_sum6 - coefficient * I_sum6;
    }

    for (int oh = H-5; oh < H; oh ++) {
        i += _W;

        // pull register (v5_~v5)
        I_v5_ = I_v4_;
        I_v4_ = I_v3_;
        I_v3_ = I_v2_;
        I_v2_ = I_v1_;
        I_v1_ = I_v;
        I_v = I_v1;
        I_v1 = I_v2;
        I_v2 = I_v3;
        I_v3 = I_v4;
        I_v4 = I_v5;  

        P_v5_ = P_v4_;
        P_v4_ = P_v3_;
        P_v3_ = P_v2_;
        P_v2_ = P_v1_;
        P_v1_ = P_v;
        P_v = P_v1;
        P_v1 = P_v2;
        P_v2 = P_v3;
        P_v3 = P_v4;
        P_v4 = P_v5;  
        //

        I_v5 = 0;
        P_v5 = 0;

        if( oh == H-5 ) {
            divisor5 = 5;
            divisor6 = 5;
        } else if( oh == H-4 ) {
            divisor5 = 4;
            divisor6 = 5;
        } else if( oh == H-3 ) {
            divisor5 = 4;
            divisor6 = 4;
        } else if( oh == H-2 ) {
            divisor5 = 3;
            divisor6 = 4;
        } else if( oh == H-1 ) {
            divisor5 = 3;
            divisor6 = 3;
        }

        I_sum5 = (I_v4_ + I_v2_ + I_v + I_v2 + I_v4) / divisor5;
        I_sum6 = (I_v5_ + I_v3_ + I_v1_ + I_v1 + I_v3) / divisor6;
        P_sum5 = (P_v4_ + P_v2_ + P_v + P_v2 + P_v4) / divisor5;
        P_sum6 = (P_v5_ + P_v3_ + P_v1_ + P_v1 + P_v3) / divisor6;

        II_sum5 = (I_v4_*I_v4_ + I_v2_*I_v2_ + I_v*I_v + I_v2*I_v2 + I_v4*I_v4) / divisor5;
        II_sum6 = (I_v5_*I_v5_ + I_v3_*I_v3_ + I_v1_*I_v1_ + I_v1*I_v1 + I_v3*I_v3) / divisor6;
        IP_sum5 = (I_v4_*P_v4_ + I_v2_*P_v2_ + I_v*P_v + I_v2*P_v2 + I_v4*P_v4) / divisor5;
        IP_sum6 = (I_v5_*P_v5_ + I_v3_*P_v3_ + I_v1_*P_v1_ + I_v1*P_v1 + I_v3*P_v3) / divisor6;

        covarianceIP = IP_sum5 - I_sum5 * P_sum5;
        varianceI = II_sum5 - I_sum5 * I_sum5;
        varianceI.x = (varianceI.x < THRESHOLD) ? THRESHOLD : varianceI.x;
        varianceI.y = (varianceI.y < THRESHOLD) ? THRESHOLD : varianceI.y;
        coefficient = covarianceIP / varianceI;
        A[i] = coefficient;
        B[i] = P_sum5 - coefficient * I_sum5;

        covarianceIP = IP_sum6 - I_sum6 * P_sum6;
        varianceI = II_sum6 - I_sum6 * I_sum6;
        varianceI.x = (varianceI.x < THRESHOLD) ? THRESHOLD : varianceI.x;
        varianceI.y = (varianceI.y < THRESHOLD) ? THRESHOLD : varianceI.y;
        coefficient = covarianceIP / varianceI;
        C[i] = coefficient;
        D[i] = P_sum6 - coefficient * I_sum6;
    }
}

__kernel void tentative_residual_vertical(__global half2 *P, __global half2 *I,
                                          __global half2 *A, __global half2 *B,
                                          __global half2 *C, __global half2 *D,
                                          __global half2 *tentative, __global half2 *residual,
                                          int H, int W) {
  int i = get_global_id(0);
  if (i*2 >= W) return;

  int _W = W/2;
  half2 A_even_sum, A_odd_sum, B_even_sum, B_odd_sum;
  half2 C_even_sum, C_odd_sum, D_even_sum, D_odd_sum;
  half2 P_pre, P_curr, P_post;
  half2 I_v;
  half2 tmp;

  int idx = i;
  A_even_sum = A[idx];
  B_even_sum = B[idx];
  C_even_sum = C[idx];
  D_even_sum = D[idx];

  idx += _W;
  A_odd_sum = A[idx];
  B_odd_sum = B[idx];
  C_odd_sum = C[idx];
  D_odd_sum = D[idx];

  idx += _W;
  A_even_sum += A[idx];
  B_even_sum += B[idx];
  C_even_sum += C[idx];
  D_even_sum += D[idx];

  idx += _W;
  A_odd_sum += A[idx];
  B_odd_sum += B[idx];
  C_odd_sum += C[idx];
  D_odd_sum += D[idx];

  idx += _W;
  A_even_sum += A[idx];
  B_even_sum += B[idx];
  C_even_sum += C[idx];
  D_even_sum += D[idx];

  idx += _W;
  A_odd_sum += A[idx];
  B_odd_sum += B[idx];
  C_odd_sum += C[idx];
  D_odd_sum += D[idx];

  P_pre = P[i];
  P_curr = P_pre;
  P_post = P[i + _W];
  I_v = (P_pre + P_post) / 2; 

  tentative[i] = (A_odd_sum+C_even_sum)/6 * P_curr + (B_odd_sum+D_even_sum)/6;
  residual[i] = P_curr - ((A_even_sum+C_odd_sum)/6 * I_v + (B_even_sum+D_odd_sum)/6);

  int divisor = 6;
  for( int oh = 1; oh < 6; oh++ ) {
    i += _W;

    idx = i + 5*_W;
    tmp = A_even_sum;
    A_even_sum = A_odd_sum;
    A_odd_sum = tmp;
    A_odd_sum += A[idx];

    idx = i + 5*_W;
    tmp = B_even_sum;
    B_even_sum = B_odd_sum;
    B_odd_sum = tmp;
    B_odd_sum += B[idx];

    idx = i + 5*_W;
    tmp = C_even_sum;
    C_even_sum = C_odd_sum;
    C_odd_sum = tmp;
    C_odd_sum += C[idx];

    idx = i + 5*_W;
    tmp = D_even_sum;
    D_even_sum = D_odd_sum;
    D_odd_sum = tmp;
    D_odd_sum += D[idx];

    divisor++;
    
    P_pre = P_curr;
    P_curr = P_post;
    P_post = P[i + _W];
    I_v = (P_pre + P_post) / 2; 

    tentative[i] = (A_odd_sum+C_even_sum)/divisor * P_curr + (B_odd_sum+D_even_sum)/divisor;
    residual[i] = P_curr - ((A_even_sum+C_odd_sum)/divisor * I_v + (B_even_sum+D_odd_sum)/divisor);
  }

  for( int oh = 6; oh < H-5; oh ++ ) {
    i += _W;
  
    idx = i + 5*_W;
    tmp = A_even_sum;
    A_even_sum = A_odd_sum;
    A_odd_sum = tmp;
    A_odd_sum += A[idx];

    idx = i - 6*_W;
    A_even_sum -= A[idx];

    idx = i + 5*_W;
    tmp = B_even_sum;
    B_even_sum = B_odd_sum;
    B_odd_sum = tmp;
    B_odd_sum += B[idx];

    idx = i - 6*_W;
    B_even_sum -= B[idx];

    idx = i + 5*_W;
    tmp = C_even_sum;
    C_even_sum = C_odd_sum;
    C_odd_sum = tmp;
    C_odd_sum += C[idx];

    idx = i - 6*_W;
    C_even_sum -= C[idx];

    idx = i + 5*_W;
    tmp = D_even_sum;
    D_even_sum = D_odd_sum;
    D_odd_sum = tmp;
    D_odd_sum += D[idx];

    idx = i - 6*_W;
    D_even_sum -= D[idx];

    P_pre = P_curr;
    P_curr = P_post;
    P_post = P[i + _W];
    I_v = (P_pre + P_post) / 2; 

    tentative[i] = (A_odd_sum+C_even_sum)/11 * P_curr + (B_odd_sum+D_even_sum)/11;
    residual[i] = P_curr - ((A_even_sum+C_odd_sum)/11 * I_v + (B_even_sum+D_odd_sum)/11);
  }

  divisor = 11;
  for( int oh=H-5; oh < H; oh++ ) {
    i += _W;

    idx = i - 6*_W;
    tmp = A_even_sum;
    A_even_sum = A_odd_sum;
    A_odd_sum = tmp;
    A_even_sum -= A[idx];

    idx = i - 6*_W;
    tmp = B_even_sum;
    B_even_sum = B_odd_sum;
    B_odd_sum = tmp;
    B_even_sum -= B[idx];

    idx = i - 6*_W;
    tmp = C_even_sum;
    C_even_sum = C_odd_sum;
    C_odd_sum = tmp;
    C_even_sum -= C[idx];

    idx = i - 6*_W;
    tmp = D_even_sum;
    D_even_sum = D_odd_sum;
    D_odd_sum = tmp;
    D_even_sum -= D[idx];

    divisor--;

    P_pre = P_curr;
    P_curr = P_post;
    P_post = (oh == H-1) ? P_post : P[i + _W];
    I_v = (P_pre + P_post) / 2; 

    tentative[i] = (A_odd_sum+C_even_sum)/divisor * P_curr + (B_odd_sum+D_even_sum)/divisor;
    residual[i] = P_curr - ((A_even_sum+C_odd_sum)/divisor * I_v + (B_even_sum+D_odd_sum)/divisor);
  }
}

__kernel void tentative_residual_horizontal(__global half *P, __global half *I,
                                            __global half4 *A, __global half4 *B,
                                            __global half4 *C, __global half4 *D,
                                            __global half *tentative,
                                            __global half *residual,
                                            int H, int W) {
  int i = get_global_id(0);
  if (i >= H * W) return;
  int oh = i / W;
  int ow = i - oh*W;

  int quiotient = i >> 2;
  int remainder = i & 0x3;
  bool remainder_is_even = ((remainder & 0x1) == 1) ? false : true;

  // vectors: A_pre2, A_pre1, A_mid, A_post1, A_post2
  half A_my_sum, A_your_sum; // coefficient A --> buffer A, C

  ///////////////////////////////////////////////////////////////// buffer A
  /////////////////////////////////////////// mid //
  half4 mid = A[quiotient];

  if (remainder_is_even) {
    A_my_sum = mid.x + mid.z;
    A_your_sum = mid.y + mid.w;
  } else {
    A_your_sum = mid.x + mid.z;
    A_my_sum = mid.y + mid.w;
  }

  /////////////////////////////////////////// pre2 //
  if (remainder == 0) {
    if (ow >= 8) {
      A_your_sum += A[quiotient - 2].w;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    half4 pre1;
    pre1 = A[quiotient - 1];

    if (remainder_is_even) {
      A_your_sum += pre1.y + pre1.w;
      A_my_sum += pre1.z;

      if (remainder == 0) {
        A_my_sum += pre1.x;
      }
    } else {
      A_your_sum += pre1.z;
      A_my_sum += pre1.w;

      if (remainder == 1) {
        A_your_sum += pre1.x;
        A_my_sum += pre1.y;
      }
    }
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    half4 post1;
    post1 = A[quiotient + 1];

    if (remainder_is_even) {
      A_my_sum += post1.x;
      A_your_sum += post1.y;

      if (remainder == 2) {
        A_my_sum += post1.z;
        A_your_sum += post1.w;
      }
    } else {
      A_your_sum += post1.x + post1.z;
      A_my_sum += post1.y;

      if (remainder == 3) {
        A_my_sum += post1.w;
      }
    }
  }

  ////////////////////////////////////////// post2 //
  if (remainder == 3) {
    if (ow < W-8) {
      A_your_sum += A[quiotient + 2].x;
    }
  }
  ///////////////////////////////////////////////////////////////// end A

  ///////////////////////////////////////////////////////////////// C
  /////////////////////////////////////////// mid //
  mid = C[quiotient];

  if (remainder_is_even) {
    A_your_sum += mid.x + mid.z;
    A_my_sum += mid.y + mid.w;
  } else {
    A_my_sum += mid.x + mid.z;
    A_your_sum += mid.y + mid.w;
  }

  /////////////////////////////////////////// pre2 //
  if (remainder == 0) {
    if (ow >= 8) {
      A_my_sum += C[quiotient - 2].w;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    half4 pre1;
    pre1 = C[quiotient - 1];

    if (remainder_is_even) {
      A_my_sum += pre1.y + pre1.w;
      A_your_sum += pre1.z;

      if (remainder == 0) {
        A_your_sum += pre1.x;
      }
    } else {
      A_my_sum += pre1.z;
      A_your_sum += pre1.w;

      if (remainder == 1) {
        A_my_sum += pre1.x;
        A_your_sum += pre1.y;
      }
    }
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    half4 post1;
    post1 = C[quiotient + 1];

    if (remainder_is_even) {
      A_your_sum += post1.x;
      A_my_sum += post1.y;

      if (remainder == 2) {
        A_your_sum += post1.z;
        A_my_sum += post1.w;
      }
    } else {
      A_my_sum += post1.x + post1.z;
      A_your_sum += post1.y;

      if (remainder == 3) {
        A_your_sum += post1.w;
      }
    }
  }

  ////////////////////////////////////////// post2 //
  if (remainder == 3) {
    if (ow < W-8) {
      A_my_sum += C[quiotient + 2].x;
    }
  }
  ///////////////////////////////////////////////////////////////// end C

  half B_my_sum, B_your_sum;  // coefficient B --> buffer B, D
  ///////////////////////////////////////////////////////////////// buffer B
  /////////////////////////////////////////// mid //
  mid = B[quiotient];

  if (remainder_is_even) {
    B_my_sum = mid.x + mid.z;
    B_your_sum = mid.y + mid.w;
  } else {
    B_your_sum = mid.x + mid.z;
    B_my_sum = mid.y + mid.w;
  }

  /////////////////////////////////////////// pre2 //
  if (remainder == 0) {
    if (ow >= 8) {
      B_your_sum += B[quiotient - 2].w;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    half4 pre1;
    pre1 = B[quiotient - 1];

    if (remainder_is_even) {
      B_your_sum += pre1.y + pre1.w;
      B_my_sum += pre1.z;

      if (remainder == 0) {
        B_my_sum += pre1.x;
      }
    } else {
      B_your_sum += pre1.z;
      B_my_sum += pre1.w;

      if (remainder == 1) {
        B_your_sum += pre1.x;
        B_my_sum += pre1.y;
      }
    }
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    half4 post1;
    post1 = B[quiotient + 1];

    if (remainder_is_even) {
      B_my_sum += post1.x;
      B_your_sum += post1.y;

      if (remainder == 2) {
        B_my_sum += post1.z;
        B_your_sum += post1.w;
      }
    } else {
      B_your_sum += post1.x + post1.z;
      B_my_sum += post1.y;

      if (remainder == 3) {
        B_my_sum += post1.w;
      }
    }
  }

  ////////////////////////////////////////// post2 //
  if (remainder == 3) {
    if (ow < W-8) {
      B_your_sum += B[quiotient + 2].x;
    }
  }
  ///////////////////////////////////////////////////////////////// end B

  ///////////////////////////////////////////////////////////////// buffer D
  /////////////////////////////////////////// mid //
  mid = D[quiotient];

  if (remainder_is_even) {
    B_your_sum += mid.x + mid.z;
    B_my_sum += mid.y + mid.w;
  } else {
    B_my_sum += mid.x + mid.z;
    B_your_sum += mid.y + mid.w;
  }

  /////////////////////////////////////////// pre2 //
  if (remainder == 0) {
    if (ow >= 8) {
      B_my_sum += D[quiotient - 2].w;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    half4 pre1;
    pre1 = D[quiotient - 1];

    if (remainder_is_even) {
      B_my_sum += pre1.y + pre1.w;
      B_your_sum += pre1.z;

      if (remainder == 0) {
        B_your_sum += pre1.x;
      }
    } else {
      B_my_sum += pre1.z;
      B_your_sum += pre1.w;

      if (remainder == 1) {
        B_my_sum += pre1.x;
        B_your_sum += pre1.y;
      }
    }
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    half4 post1;
    post1 = D[quiotient + 1];

    if (remainder_is_even) {
      B_your_sum += post1.x;
      B_my_sum += post1.y;

      if (remainder == 2) {
        B_your_sum += post1.z;
        B_my_sum += post1.w;
      }
    } else {
      B_my_sum += post1.x + post1.z;
      B_your_sum += post1.y;

      if (remainder == 3) {
        B_your_sum += post1.w;
      }
    }
  }

  ////////////////////////////////////////// post2 //
  if (remainder == 3) {
    if (ow < W-8) {
      B_my_sum += D[quiotient + 2].x;
    }
  }
  ///////////////////////////////////////////////////////////////// end D

  int divisor;
  if (ow < 5)
    divisor = (6 + ow);
  else if (ow > W-6)
    divisor = (W + 5 - ow);
  else
    divisor = 11;

  A_my_sum = A_my_sum / divisor;
  B_my_sum = B_my_sum / divisor;
  A_your_sum = A_your_sum / divisor;
  B_your_sum = B_your_sum / divisor;

  half valueP, valueI;
  {
    unsigned char rem = (ow & 0x1);
    half2 Pv = vload2(i >> 1, P);

    if (rem == 0) {
      valueP = Pv.x;
      valueI = Pv.y;

      if (ow > 1) {
        Pv = vload2((i-1) >> 1, P);
        valueI += Pv.y;
      }
      else {
        valueI += Pv.x;
      }
      valueI /= 2;
    }
    else {
      valueP = Pv.y;
      valueI = Pv.x;

      if (ow < W-2) {
        Pv = vload2((i+1) >> 1, P);
        valueI += Pv.x;
      }
      else {
        valueI += Pv.y;
      }
      valueI /= 2;
    }
  }
  tentative[i] = A_your_sum * valueP + B_your_sum;
  residual[i] = valueP - (A_my_sum * valueI + B_my_sum);
}

__kernel void residual_interpolation(__global half *P,
                                     __global half *residual_H, __global half *residual_V,
                                     __global half *tentative_H, __global half *tentative_V,
                                     int H, int W) {
  int i = get_global_id(0);
  if( i >= H*W ) return;

  int oh = i/W;
  int ow = i%W;

  half tenta_H, tenta_V;

  i = oh*W+ow;

  int ii;
  // residual intepolation
  // filter : {1/2, 0, 1/2}

  // for horizontal 
  ii = ow-1;
  if ( ii < 0 ) tenta_H = 0;
  else tenta_H = residual_H[i-1];

  ii = ow+1;
  if ( ii >= W ) tenta_H += 0;
  else tenta_H += residual_H[i+1];
  tenta_H = tenta_H / 2;

  // for vertical
  ii = oh-1;
  if ( ii < 0 ) tenta_V = 0;
  else tenta_V = residual_V[ii*W+ow];
  ii = oh+1;
  if ( ii >= H ) tenta_V += 0;
  else tenta_V += residual_V[ii*W+ow];
  tenta_V = tenta_V / 2;

  // tentative + residual
  tenta_H = tentative_H[i] + tenta_H;
  tenta_V = tentative_V[i] + tenta_V;

  // diff = raw - (tentative + residual)
  //여기 바꿔야할듯
  int oh_last_bit = oh & 0x1;
  int ow_last_bit = ow & 0x1;
  half valueP = P[i];

  if( oh_last_bit == 0 ) {
    if( ow_last_bit == 0 ) {
      tenta_H = valueP - tenta_H;
      tenta_V = valueP - tenta_V;
    } else {
      tenta_H = tenta_H - valueP;
      tenta_V = tenta_V - valueP;
    }
  } else {
    if( ow_last_bit == 0 ) {
      tenta_H = tenta_H - valueP;
      tenta_V = tenta_V - valueP;
    } else {
      tenta_H = valueP - tenta_H;
      tenta_V = valueP - tenta_V;
    }
  }

  tentative_H[i] = tenta_H;
  tentative_V[i] = tenta_V;
}

__kernel void color_difference_h(__global half4 *difh, __global half *difh2,
                                 __global half  *difw, __global half *dife,
                                 int  H, int W) {
  int i = get_global_id(0);
  if (i >= H*W) return;

  int oh = i/W;
  int ow = i%W;

  int quiotient = i / 4;
  int remainder = i - quiotient * 4;
  // vectors: pre_1, mid, post_1

  half diff;
  half difw_sum, dife_sum;

  /////////////////////////////////////////// mid //
  half4 mid = difh[quiotient];
  switch (remainder) {
    half mid_v;
    case 0: 
    mid_v = mid.x*WT;
    difw_sum = mid_v;
    dife_sum = mid_v + mid.y*WT1 + mid.z*WT2 + mid.w*WT3; 
    diff = -mid.y;
    break;
    case 1: 
    mid_v = mid.y*WT;
    difw_sum = mid.x*WT1 + mid_v;
    dife_sum = mid_v + mid.z*WT1 + mid.w*WT2; 
    diff = mid.x - mid.z;
    break;
    case 2: 
    mid_v = mid.z*WT;
    difw_sum = mid.x*WT2 + mid.y*WT1 + mid_v;
    dife_sum = mid_v + mid.w*WT1; 
    diff = mid.y - mid.w;
    break;
    case 3: 
    mid_v = mid.w*WT;
    difw_sum = mid.x*WT3 + mid.y*WT2 + mid.z*WT1 + mid_v;
    dife_sum = mid_v; 
    diff = mid.z;
    break;
  }
  half v;
  switch (remainder) {
    case 0: v = mid.x; break;
    case 1: v = mid.y; break;
    case 2: v = mid.z; break;
    case 3: v = mid.w; break;
  }
  /////////////////////////////////////////// pre1 //
  half4 pre1;
  if (ow < 4) {
    pre1 = (half4)(mid.x);
  } else {
    pre1 = difh[quiotient - 1];
  }

  switch (remainder) {
    case 0: 
      difw_sum += pre1.x*WT4 + pre1.y*WT3 + pre1.z*WT2 + pre1.w*WT1; 
      diff += pre1.w;
      break;
    case 1: 
      difw_sum += pre1.y*WT4 + pre1.z*WT3 + pre1.w*WT2; 
      break;
    case 2: 
      difw_sum += pre1.z*WT4 + pre1.w*WT3;
      break;
    case 3: 
      difw_sum += pre1.w*WT4; 
      break;
  }
  /////////////////////////////////////////// post1 //
  half4 post1;
  if (ow > W-5 ) {
    post1 = (half4)(mid.w);
  } else {
    post1 = difh[quiotient + 1];
  }

  switch (remainder) {
    case 0: 
      dife_sum += post1.x*WT4; 
      break;
    case 1: 
      dife_sum += post1.x*WT3 + post1.y*WT4; 
      break;
    case 2: 
      dife_sum += post1.x*WT2 + post1.y*WT3 + post1.z*WT4;
      break;
    case 3: 
      dife_sum += post1.x*WT1 + post1.y*WT2 + post1.z*WT3 + post1.w*WT4; 
      diff -= post1.x;
      break;
  }

  // output
  difw[i] = difw_sum;
  dife[i] = dife_sum;
  difh2[i] = fabs(diff);
}

__kernel void color_difference_v(__global half *difv, __global half *difv2,
                                 __global half  *difn, __global half *difs,
                                 int  H, int W) {
  int ow = get_global_id(0);
  int oh = get_global_id(1);
  if( ow >= W || oh >= H ) return;
  int i = oh*W+ow;
  int ii;
  half x;

  /////////////////////////////////// vertical : difv2

  half v = difv[i];
  ii = oh-1;
  if( ii < 0 ) x = v;
  else x = difv[ii*W+ow];

  ii = oh+1;
  if( ii >= H ) x -= v;
  else x -= difv[ii*W+ow];

  difv2[i] = fabs(x);

  /////////////////////////////////// difn, difs
  half b;
  if( oh < 4 ) b = difv[ow];

  x = v*0.570350;
  ii = oh-1;
  if( ii < 0 ) x += b*0.345934;
  else x += difv[ii*W+ow]*0.345934;

  ii = oh-2;
  if( ii < 0 ) x += b*0.077188;
  else x += difv[ii*W+ow]*0.077188;

  ii = oh-3;
  if( ii < 0 ) x += b*0.006336;
  else x += difv[ii*W+ow]*0.006336;

  ii = oh-4;
  if( ii < 0 ) x += b*0.000192;
  else x += difv[ii*W+ow]*0.000192;

  difn[i] = x;

  if( oh > H-5 ) b = difv[(H-1)*W+ow];

  x = v*0.570350;
  ii = oh+1;
  if( ii >= H ) x += b*0.345934;
  else x += difv[ii*W+ow]*0.345934;

  ii = oh+2;
  if( ii >= H ) x += b*0.077188;
  else x += difv[ii*W+ow]*0.077188;

  ii = oh+3;
  if( ii >= H ) x += b*0.006336;
  else x += difv[ii*W+ow]*0.006336;

  ii = oh+4;
  if( ii >= H ) x += b*0.000192;
  else x += difv[ii*W+ow]*0.000192;

  difs[i] = x;
}

__kernel void compute_weight_new(__global half *in, __global half *out,
                                 int  H, int W) {
  int ow = get_global_id(0) * 2;
  int oh = get_global_id(1) * 2;
  if (ow >= W || oh >= H) return;

  unsigned int index;

  half4 val;

  half sum_00, sum_01;
  half sum_10, sum_11;
  sum_00 = sum_01 = sum_10 = sum_11 = 0;

  /////////////////////////////////// h-2
  if (oh < 2) {
    ;
  }
  else {
    if (ow < 2) {
      val = (half4)0;
    }
    else {
      index = (oh-2)*W + (ow - 2);
      val = vload4(index/4, in);
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.z + val.w;
      sum_01 += val.w;
    }
    else {
      sum_00 += val.x + val.y + val.z + val.w;
      sum_01 += val.y + val.z + val.w;
    }

    if (ow < W-2) {
      index = (oh-2)*W + (ow + 2);
      val = vload4(index/4, in);
    }
    else {
      val = (half4)0;
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.x + val.y + val.z;
      sum_01 += val.x + val.y + val.z + val.w;
    }
    else {
      sum_00 += val.x;
      sum_01 += val.x + val.y;
    }
  }

  /////////////////////////////////// h-1
  if (oh < 1) {
    ;
  }
  else {
    if (ow < 2) {
      val = (half4)0;
    }
    else {
      index = (oh-1)*W + (ow - 2);
      val = vload4(index/4, in);
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.z + val.w;
      sum_01 += val.w;
      sum_10 += val.z + val.w;
      sum_11 += val.w;
    }
    else {
      sum_00 += val.x + val.y + val.z + val.w;
      sum_01 += val.y + val.z + val.w;
      sum_10 += val.x + val.y + val.z + val.w;
      sum_11 += val.y + val.z + val.w;
    }

    if (ow < W-2) {
      index = (oh-1)*W + (ow + 2);
      val = vload4(index/4, in);
    }
    else {
      val = (half4)0;
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.x + val.y + val.z;
      sum_01 += val.x + val.y + val.z + val.w;
      sum_10 += val.x + val.y + val.z;
      sum_11 += val.x + val.y + val.z + val.w;
    }
    else {
      sum_00 += val.x;
      sum_01 += val.x + val.y;
      sum_10 += val.x;
      sum_11 += val.x + val.y;
    }
  }

  /////////////////////////////////// h+0
  {
    if (ow < 2) {
      val = (half4)0;
    }
    else {
      index = (oh)*W + (ow - 2);
      val = vload4(index/4, in);
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.z + val.w;
      sum_01 += val.w;
      sum_10 += val.z + val.w;
      sum_11 += val.w;
    }
    else {
      sum_00 += val.x + val.y + val.z + val.w;
      sum_01 += val.y + val.z + val.w;
      sum_10 += val.x + val.y + val.z + val.w;
      sum_11 += val.y + val.z + val.w;
    }

    if (ow < W-2) {
      index = (oh)*W + (ow + 2);
      val = vload4(index/4, in);
    }
    else {
      val = (half4)0;
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.x + val.y + val.z;
      sum_01 += val.x + val.y + val.z + val.w;
      sum_10 += val.x + val.y + val.z;
      sum_11 += val.x + val.y + val.z + val.w;
    }
    else {
      sum_00 += val.x;
      sum_01 += val.x + val.y;
      sum_10 += val.x;
      sum_11 += val.x + val.y;
    }
  }

  /////////////////////////////////// h+1
  if (oh < H-1) {
    if (ow < 2) {
      val = (half4)0;
    }
    else {
      index = (oh+1)*W + (ow - 2);
      val = vload4(index/4, in);
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.z + val.w;
      sum_01 += val.w;
      sum_10 += val.z + val.w;
      sum_11 += val.w;
    }
    else {
      sum_00 += val.x + val.y + val.z + val.w;
      sum_01 += val.y + val.z + val.w;
      sum_10 += val.x + val.y + val.z + val.w;
      sum_11 += val.y + val.z + val.w;
    }

    if (ow < W-2) {
      index = (oh+1)*W + (ow + 2);
      val = vload4(index/4, in);
    }
    else {
      val = (half4)0;
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.x + val.y + val.z;
      sum_01 += val.x + val.y + val.z + val.w;
      sum_10 += val.x + val.y + val.z;
      sum_11 += val.x + val.y + val.z + val.w;
    }
    else {
      sum_00 += val.x;
      sum_01 += val.x + val.y;
      sum_10 += val.x;
      sum_11 += val.x + val.y;
    }
  }

  /////////////////////////////////// h+2
  if (oh < H-2) {
    if (ow < 2) {
      val = (half4)0;
    }
    else {
      index = (oh+2)*W + (ow - 2);
      val = vload4(index/4, in);
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.z + val.w;
      sum_01 += val.w;
      sum_10 += val.z + val.w;
      sum_11 += val.w;
    }
    else {
      sum_00 += val.x + val.y + val.z + val.w;
      sum_01 += val.y + val.z + val.w;
      sum_10 += val.x + val.y + val.z + val.w;
      sum_11 += val.y + val.z + val.w;
    }

    if (ow < W-2) {
      index = (oh+2)*W + (ow + 2);
      val = vload4(index/4, in);
    }
    else {
      val = (half4)0;
    }
    if ((ow & 0x2) == 0) {
      sum_00 += val.x + val.y + val.z;
      sum_01 += val.x + val.y + val.z + val.w;
      sum_10 += val.x + val.y + val.z;
      sum_11 += val.x + val.y + val.z + val.w;
    }
    else {
      sum_00 += val.x;
      sum_01 += val.x + val.y;
      sum_10 += val.x;
      sum_11 += val.x + val.y;
    }
  }

  /////////////////////////////////// h+3
  if (oh < H-3) {
    if (ow < 2) {
      val = (half4)0;
    }
    else {
      index = (oh+3)*W + (ow - 2);
      val = vload4(index/4, in);
    }
    if ((ow & 0x2) == 0) {
      sum_10 += val.z + val.w;
      sum_11 += val.w;
    }
    else {
      sum_10 += val.x + val.y + val.z + val.w;
      sum_11 += val.y + val.z + val.w;
    }

    if (ow < W-2) {
      index = (oh+3)*W + (ow + 2);
      val = vload4(index/4, in);
    }
    else {
      val = (half4)0;
    }
    if ((ow & 0x2) == 0) {
      sum_10 += val.x + val.y + val.z;
      sum_11 += val.x + val.y + val.z + val.w;
    }
    else {
      sum_10 += val.x;
      sum_11 += val.x + val.y;
    }
  }

  vstore2((half2)(sum_00, sum_01), ((oh+0)*W+ow)/2, out);
  vstore2((half2)(sum_10, sum_11), ((oh+1)*W+ow)/2, out);
}

__kernel void directional_weight(__global half *raw,
                                 __global half *difn, __global half *difs,
                                 __global half *difw, __global half *dife,
                                 __global half *wh, __global half  *wv,
                                 __global half *green, int  H, int W) {
  int i = get_global_id(0);
  if( i >= H*W ) return;

  int oh = i/W;
  int ow = i%W;

  half v = wh[i];

  half ww, we, wn, ws, wt;

  int ii = ow-2;
  if( ii < 0 ) ii = 0;
  ww = wh[oh*W+ii];
  ww = 1.0 / (ww * ww + exp10(-32.0));

  ii = ow+2;
  if( ii >= W ) ii = W-1;
  we = wh[oh*W+ii];
  we = 1.0 / (we * we + exp10(-32.0));

  ii = oh-2;
  if( ii < 0 ) ii = 0;
  wn = wv[ii*W+ow];
  wn = 1.0 / (wn * wn + exp10(-32.0));

  ii = oh+2;
  if( ii >= H ) ii = H-1; 
  ws = wv[ii*W+ow];
  ws = 1.0 / (ws * ws + exp10(-32.0));

  wt = ww + we + wn + ws;

  // dif = (Wn * difn + Ws * difs + Ww * difw + We * dife) / wt;
  wt = (wn * difn[i] + ws * difs[i] + ww * difw[i] + we * dife[i]) / wt;

  int oh_last_bit = oh & 0x1;
  int ow_last_bit = ow & 0x1;
  v = raw[i];
  if( (oh_last_bit==0 && ow_last_bit==0) || (oh_last_bit==1 && ow_last_bit==1) ) { // green channel
    wt = v; 
  } else { // not green channel
    wt = wt + v;

    // crop 0~31
    if (wt < 0) wt = 0;
    else if (wt > 31) wt = 31;
  }

  green[i] = wt;
}
