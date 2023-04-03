#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define THRESHOLD  0.65025

__kernel void red_blue_vertical_sum(__global half2 *I, __global half2 *P,
                                    __global half2 *sumI, __global half2 *sumP,
                                    __global half2 *sumII, __global half2 *sumIP,
                                    int H, int W) {
  int i = get_global_id(0);
  if (i*2 >= W) return;

  half2 I_v5_=0, I_v4_=0, I_v3_=0, I_v2_=0, I_v1_=0;
  half2 I_v, I_v1, I_v2, I_v3, I_v4, I_v5;
  half2 P_v5_=0, P_v4_=0, P_v3_=0, P_v2_=0, P_v1_=0;
  half2 P_v, P_v1, P_v2, P_v3, P_v4, P_v5, P_v6;
  half2 I_sum5, I_sum6, P_sum5, P_sum6, II_sum5, II_sum6, IP_sum5, IP_sum6;

  int _W = W/2;

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

  I_sum5 = I_v + I_v2 + I_v4;
  I_sum6 = I_v1 + I_v3 + I_v5;
  P_sum5 = P_v + P_v2 + P_v4;;  
  P_sum6 = P_v1 + P_v3 + P_v5;

  II_sum5 = I_v*I_v + I_v2*I_v2 + I_v4*I_v4;
  II_sum6 = I_v1*I_v1 + I_v3*I_v3 + I_v5*I_v5;
  IP_sum5 = I_v*P_v + I_v2*P_v2 + I_v4*P_v4;
  IP_sum6 = I_v1*P_v1 + I_v3*P_v3 + I_v5*P_v5;

  half2 result;
  result.x = I_sum6.x;
  result.y = I_sum5.y;
  sumI[i] = result;

  result.x = P_sum6.x;
  result.y = P_sum5.y;
  sumP[i] = result;

  result.x = II_sum6.x;
  result.y = II_sum5.y;
  sumII[i] = result;

  result.x = IP_sum6.x;
  result.y = IP_sum5.y;
  sumIP[i] = result;

  //////////////////////////////////////////////////// next sum
  for (int oh = 1; oh < 5; oh ++) {
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

    I_sum5 = I_v4_ + I_v2_ + I_v + I_v2 + I_v4;
    I_sum6 = I_v5_ + I_v3_ + I_v1_ + I_v1 + I_v3 + I_v5;
    P_sum5 = P_v4_ + P_v2_ + P_v + P_v2 + P_v4;
    P_sum6 = P_v5_ + P_v3_ + P_v1_ + P_v1 + P_v3 + P_v5;

    II_sum5 = I_v4_*I_v4_ + I_v2_*I_v2_ + I_v*I_v + I_v2*I_v2 + I_v4*I_v4;
    II_sum6 = I_v5_*I_v5_ + I_v3_*I_v3_ + I_v1_*I_v1_ + I_v1*I_v1 + I_v3*I_v3 + I_v5*I_v5;
    IP_sum5 = I_v4_*P_v4_ + I_v2_*P_v2_ + I_v*P_v + I_v2*P_v2 + I_v4*P_v4;
    IP_sum6 = I_v5_*P_v5_ + I_v3_*P_v3_ + I_v1_*P_v1_ + I_v1*P_v1 + I_v3*P_v3 + I_v5*P_v5;

    if( (oh & 0x1) == 0 ) {
      result.x = I_sum6.x;
      result.y = I_sum5.y;
      sumI[i] = result;

      result.x = P_sum6.x;
      result.y = P_sum5.y;
      sumP[i] = result;

      result.x = II_sum6.x;
      result.y = II_sum5.y;
      sumII[i] = result;

      result.x = IP_sum6.x;
      result.y = IP_sum5.y;
      sumIP[i] = result;
    } else {
      result.x = I_sum5.x;
      result.y = I_sum6.y;
      sumI[i] = result;

      result.x = P_sum5.x;
      result.y = P_sum6.y;
      sumP[i] = result;

      result.x = II_sum5.x;
      result.y = II_sum6.y;
      sumII[i] = result;

      result.x = IP_sum5.x;
      result.y = IP_sum6.y;
      sumIP[i] = result;
    }
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

    I_sum5 = I_v4_ + I_v2_ + I_v + I_v2 + I_v4;
    I_sum6 = I_v5_ + I_v3_ + I_v1_ + I_v1 + I_v3 + I_v5;
    P_sum5 = P_v4_ + P_v2_ + P_v + P_v2 + P_v4;
    P_sum6 = P_v5_ + P_v3_ + P_v1_ + P_v1 + P_v3 + P_v5;

    II_sum5 = I_v4_*I_v4_ + I_v2_*I_v2_ + I_v*I_v + I_v2*I_v2 + I_v4*I_v4;
    II_sum6 = I_v5_*I_v5_ + I_v3_*I_v3_ + I_v1_*I_v1_ + I_v1*I_v1 + I_v3*I_v3 + I_v5*I_v5;
    IP_sum5 = I_v4_*P_v4_ + I_v2_*P_v2_ + I_v*P_v + I_v2*P_v2 + I_v4*P_v4;
    IP_sum6 = I_v5_*P_v5_ + I_v3_*P_v3_ + I_v1_*P_v1_ + I_v1*P_v1 + I_v3*P_v3 + I_v5*P_v5;

    if( (oh & 0x1) == 0 ) {
      result.x = I_sum6.x;
      result.y = I_sum5.y;
      sumI[i] = result;

      result.x = P_sum6.x;
      result.y = P_sum5.y;
      sumP[i] = result;

      result.x = II_sum6.x;
      result.y = II_sum5.y;
      sumII[i] = result;

      result.x = IP_sum6.x;
      result.y = IP_sum5.y;
      sumIP[i] = result;
    } else {
      result.x = I_sum5.x;
      result.y = I_sum6.y;
      sumI[i] = result;

      result.x = P_sum5.x;
      result.y = P_sum6.y;
      sumP[i] = result;

      result.x = II_sum5.x;
      result.y = II_sum6.y;
      sumII[i] = result;

      result.x = IP_sum5.x;
      result.y = IP_sum6.y;
      sumIP[i] = result;
    }
  }

  for( int oh = H-5; oh < H; oh ++ ) {
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

    I_v5 = 0;
    P_v5 = 0;

    I_sum5 = I_v4_ + I_v2_ + I_v + I_v2 + I_v4;
    I_sum6 = I_v5_ + I_v3_ + I_v1_ + I_v1 + I_v3;
    P_sum5 = P_v4_ + P_v2_ + P_v + P_v2 + P_v4;
    P_sum6 = P_v5_ + P_v3_ + P_v1_ + P_v1 + P_v3;

    II_sum5 = I_v4_*I_v4_ + I_v2_*I_v2_ + I_v*I_v + I_v2*I_v2 + I_v4*I_v4;
    II_sum6 = I_v5_*I_v5_ + I_v3_*I_v3_ + I_v1_*I_v1_ + I_v1*I_v1 + I_v3*I_v3;
    IP_sum5 = I_v4_*P_v4_ + I_v2_*P_v2_ + I_v*P_v + I_v2*P_v2 + I_v4*P_v4;
    IP_sum6 = I_v5_*P_v5_ + I_v3_*P_v3_ + I_v1_*P_v1_ + I_v1*P_v1 + I_v3*P_v3;

    if( (oh & 0x1) == 0 ) {
      result.x = I_sum6.x;
      result.y = I_sum5.y;
      sumI[i] = result;

      result.x = P_sum6.x;
      result.y = P_sum5.y;
      sumP[i] = result;

      result.x = II_sum6.x;
      result.y = II_sum5.y;
      sumII[i] = result;

      result.x = IP_sum6.x;
      result.y = IP_sum5.y;
      sumIP[i] = result;
    } else {
      result.x = I_sum5.x;
      result.y = I_sum6.y;
      sumI[i] = result;

      result.x = P_sum5.x;
      result.y = P_sum6.y;
      sumP[i] = result;

      result.x = II_sum5.x;
      result.y = II_sum6.y;
      sumII[i] = result;

      result.x = IP_sum5.x;
      result.y = IP_sum6.y;
      sumIP[i] = result;
    }
  }
}

__kernel void red_blue_residual_interpolation_and_add_tentative(__global half *residualRB,
                                                                __global half *tentativeR,
                                                                __global half *tentativeB,
                                                                int H, int W) {
  // work register : 13, uniform register : 20
  int i = get_global_id(0);
  if (i >= H * W) return; // boundary check

  int oh = i / W;
  int ow = i % W;

  int iw, ih, flag;
  half value_red=0, value_blue=0;

  int ow_last_bit = ow & 0x1;
  int oh_last_bit = oh & 0x1;

  if( oh_last_bit == 0 && ow_last_bit == 0 ) flag = 0; // GR
  else if( oh_last_bit == 0 && ow_last_bit == 1 ) flag = 1; // R
  else if( oh_last_bit == 1 && ow_last_bit == 0 ) flag = 2; // B
  else flag = 3; // GB

  if( flag == 0 ) {
    iw = ow-1;
    if ( iw < 0 ) 
      value_red += 0;
    else 
      value_red += residualRB[oh*W+iw];

    iw = ow+1;
    if ( iw >= W) 
      value_red += 0;
    else 
      value_red += residualRB[oh*W+iw];

    value_red  = value_red/2;

    ih = oh-1;
    if ( ih < 0 )
      value_blue += 0;
    else 
      value_blue += residualRB[ih*W+ow];

    ih = oh+1;
    if ( ih >= H )
      value_blue += 0;
    else 
      value_blue += residualRB[ih*W+ow];

    value_blue  = value_blue/2;

  } else if( flag == 1 ) {

    value_red = residualRB[i];

    iw = ow-1;
    ih = oh-1;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) 
      value_blue += 0;
    else 
      value_blue += residualRB[ih*W+iw];

    iw = ow+1;
    ih = oh-1;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) 
      value_blue += 0;
    else 
      value_blue += residualRB[ih*W+iw];

    iw = ow-1;
    ih = oh+1;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) 
      value_blue += 0;
    else 
      value_blue += residualRB[ih*W+iw];

    iw = ow+1;
    ih = oh+1;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) 
      value_blue += 0;
    else 
      value_blue += residualRB[ih*W+iw];

    value_blue = value_blue / 4;

  } else if( flag == 2 ) {

    value_blue = residualRB[i];

    iw = ow-1;
    ih = oh-1;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) 
      value_red += 0;
    else 
      value_red += residualRB[ih*W+iw];

    iw = ow+1;
    ih = oh-1;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) 
      value_red += 0;
    else 
      value_red += residualRB[ih*W+iw];

    iw = ow-1;
    ih = oh+1;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) 
      value_red += 0;
    else 
      value_red += residualRB[ih*W+iw];

    iw = ow+1;
    ih = oh+1;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) 
      value_red += 0;
    else 
      value_red += residualRB[ih*W+iw];

    value_red = value_red / 4;

  } else {

    iw = ow-1;
    if ( iw < 0 ) 
      value_blue += 0;
    else 
      value_blue += residualRB[oh*W+iw];

    iw = ow+1;
    if ( iw >= W) 
      value_blue += 0;
    else 
      value_blue += residualRB[oh*W+iw];

    value_blue  = value_blue/2;

    ih = oh-1;
    if ( ih < 0 )
      value_red += 0;
    else 
      value_red += residualRB[ih*W+ow];

    ih = oh+1;
    if ( ih >= H )
      value_red += 0;
    else 
      value_red += residualRB[ih*W+ow];

    value_red  = value_red/2;
  }

  tentativeR[i] += value_red;
  tentativeB[i] += value_blue;
}

__kernel void horizontal_sum_and_linear_coefficient(__global half4 *I, __global half4 *P,
                                                    __global half4 *II, __global half4 *IP,
                                                    __global half *red_coa,
                                                    __global half *red_cob,
                                                    __global half *blue_coa,
                                                    __global half *blue_cob,
                                                    int H, int W) {
  int i = get_global_id(0);
  if (i >= H * W) return;
  int oh = i / W;
  int ow = i - oh*W;

  int quiotient = i >> 2;
  int remainder = i & 0x3;

  // vectors: pre2, pre1, mid, post1, post2 (I, II, IP, P)
  half4 pre1, pre2, post1, post2;

  half I_even_sum, I_odd_sum, P_even_sum, P_odd_sum;
  half II_even_sum, II_odd_sum, IP_even_sum, IP_odd_sum;

  bool remainder_is_even = ((remainder & 0x1) == 1) ? false : true;

  ///////////////////////////////////////////////////////////////// buffer I
  /////////////////////////////////////////// mid //
  half4 mid = I[quiotient];

  if (remainder_is_even) {
    I_even_sum = mid.x + mid.z;
    I_odd_sum = mid.y + mid.w;
  } else {
    I_odd_sum = mid.x + mid.z;
    I_even_sum = mid.y + mid.w;
  }

  /////////////////////////////////////////// pre2 //
  if (remainder == 0) {
    if (ow >= 8) {
      I_odd_sum += I[quiotient - 2].w;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    pre1 = I[quiotient - 1];

    if (remainder_is_even) {
      I_odd_sum += pre1.y + pre1.w;
      I_even_sum += pre1.z;

      if (remainder == 0) {
        I_even_sum += pre1.x;
      }
    } else {
      I_odd_sum += pre1.z;
      I_even_sum += pre1.w;

      if (remainder == 1) {
        I_odd_sum += pre1.x;
        I_even_sum += pre1.y;
      }
    }
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    post1 = I[quiotient + 1];

    if (remainder_is_even) {
      I_even_sum += post1.x;
      I_odd_sum += post1.y;

      if(remainder == 2) {
        I_even_sum += post1.z;
        I_odd_sum += post1.w;
      }
    } else {
      I_odd_sum += post1.x + post1.z;
      I_even_sum += post1.y;

      if (remainder == 3) {
        I_even_sum += post1.w;
      }
    }
  }

  ////////////////////////////////////////// post2 //
  if (remainder == 3) {
    if (ow < W-8) {
      I_odd_sum += I[quiotient + 2].x;
    }
  }
  ///////////////////////////////////////////////////////////////// end I


  ///////////////////////////////////////////////////////////////// buffer P
  /////////////////////////////////////////// mid //
  mid = P[quiotient];

  if (remainder_is_even) {
    P_even_sum = mid.x + mid.z;
    P_odd_sum = mid.y + mid.w;
  } else {
    P_odd_sum = mid.x + mid.z;
    P_even_sum = mid.y + mid.w;
  }

  /////////////////////////////////////////// pre2 //
  if (remainder == 0) {
    if (ow >= 8) {
      P_odd_sum += P[quiotient - 2].w;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    pre1 = P[quiotient - 1];

    if (remainder_is_even) {
      P_odd_sum += pre1.y + pre1.w;
      P_even_sum += pre1.z;

      if (remainder == 0) {
        P_even_sum += pre1.x;
      }
    } else {
      P_odd_sum += pre1.z;
      P_even_sum += pre1.w;

      if (remainder == 1) {
        P_odd_sum += pre1.x;
        P_even_sum += pre1.y;
      }
    }
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    post1 = P[quiotient + 1];

    if (remainder_is_even) {
      P_even_sum += post1.x;
      P_odd_sum += post1.y;

      if (remainder == 2) {
        P_even_sum += post1.z;
        P_odd_sum += post1.w;
      }
    } else {
      P_odd_sum += post1.x + post1.z;
      P_even_sum += post1.y;

      if (remainder == 3) {
        P_even_sum += post1.w;
      }
    }
  }

  ////////////////////////////////////////// post2 //
  if (remainder == 3) {
    if (ow < W-8) {
      P_odd_sum += P[quiotient + 2].x;
    }
  }
  ///////////////////////////////////////////////////////////////// end P

  ///////////////////////////////////////////////////////////////// buffer II
  /////////////////////////////////////////// mid //
  mid = II[quiotient];

  if (remainder_is_even) {
    II_even_sum = mid.x + mid.z;
    II_odd_sum = mid.y + mid.w;
  } else {
    II_odd_sum = mid.x + mid.z;
    II_even_sum = mid.y + mid.w;
  }

  /////////////////////////////////////////// pre2 //
  if (remainder == 0) {
    if (ow >= 8) {
      II_odd_sum += II[quiotient - 2].w;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    pre1 = II[quiotient - 1];

    if (remainder_is_even) {
      II_odd_sum += pre1.y + pre1.w;
      II_even_sum += pre1.z;

      if (remainder == 0) {
        II_even_sum += pre1.x;
      }
    } else {
      II_odd_sum += pre1.z;
      II_even_sum += pre1.w;

      if (remainder == 1) {
        II_odd_sum += pre1.x;
        II_even_sum += pre1.y;
      }
    }
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    post1 = II[quiotient + 1];

    if (remainder_is_even) {
      II_even_sum += post1.x;
      II_odd_sum += post1.y;

      if (remainder == 2) {
        II_even_sum += post1.z;
        II_odd_sum += post1.w;
      }
    } else {
      II_odd_sum += post1.x + post1.z;
      II_even_sum += post1.y;

      if (remainder == 3) {
        II_even_sum += post1.w;
      }
    }
  }

  ////////////////////////////////////////// post2 //
  if (remainder == 3) {
    if (ow < W-8) {
      II_odd_sum += II[quiotient + 2].x;
    }
  }
  ///////////////////////////////////////////////////////////////// end II

  ///////////////////////////////////////////////////////////////// buffer IP
  /////////////////////////////////////////// mid //
  mid = IP[quiotient];

  if (remainder_is_even) {
    IP_even_sum = mid.x + mid.z;
    IP_odd_sum = mid.y + mid.w;
  } else {
    IP_odd_sum = mid.x + mid.z;
    IP_even_sum = mid.y + mid.w;
  }

  /////////////////////////////////////////// pre2 //
  if (remainder == 0) {
    if (ow >= 8) {
      IP_odd_sum += IP[quiotient - 2].w;
    }
  }

  /////////////////////////////////////////// pre1 //
  if (ow >= 4) {
    pre1 = IP[quiotient - 1];

    if (remainder_is_even) {
      IP_odd_sum += pre1.y + pre1.w;
      IP_even_sum += pre1.z;

      if (remainder == 0) {
        IP_even_sum += pre1.x;
      }
    } else {
      IP_odd_sum += pre1.z;
      IP_even_sum += pre1.w;

      if (remainder == 1) {
        IP_odd_sum += pre1.x;
        IP_even_sum += pre1.y;
      }
    }
  }

  ////////////////////////////////////////// post1 //
  if (ow < W-4) {
    post1 = IP[quiotient + 1];

    if (remainder_is_even) {
      IP_even_sum += post1.x;
      IP_odd_sum += post1.y;

      if (remainder == 2) {
        IP_even_sum += post1.z;
        IP_odd_sum += post1.w;
      }
    } else {
      IP_odd_sum += post1.x + post1.z;
      IP_even_sum += post1.y;

      if (remainder == 3) {
        IP_even_sum += post1.w;
      }
    }
  }
  ////////////////////////////////////////////////////
  ////////////////////////////////////////// post2 //
  if (remainder == 3) {
    if (ow < W-8) {
      IP_odd_sum += IP[quiotient + 2].x;
    }
  }
  ///////////////////////////////////////////////////////////////// end IP

  ////////////////////////////////////////// divisor calculation
  int divisor_even_h, divisor_odd_h;
  if (ow < 5) {
    divisor_even_h = 3 + (ow >> 1);
    divisor_odd_h = 3 + ((ow + 1) >> 1);
  }
  else if (ow > W-6) {
    divisor_even_h = 3 + ((W - ow - 1) >> 1);
    divisor_odd_h = 3 + ((W - ow) >> 1);
  }
  else {
    divisor_even_h = 5;
    divisor_odd_h = 6;
  }

  int divisor_even_v, divisor_odd_v;
  if (oh < 5) {
    divisor_even_v = 3 + (oh >> 1);
    divisor_odd_v = 3 + ((oh + 1) >> 1);
  }
  else if (oh > H-6) {
    divisor_even_v = 3 + ((H - oh - 1) >> 1);
    divisor_odd_v = 3 + ((H - oh) >> 1);
  }
  else {
    divisor_even_v = 5;
    divisor_odd_v = 6;
  }

  int oh_last_bit = oh & 0x1;
  int ow_last_bit = ow & 0x1;
  int divisor_even, divisor_odd;
  if( ((oh_last_bit == 0) && (ow_last_bit == 1)) ||
      ((oh_last_bit == 1) && (ow_last_bit == 0)) ) {
    divisor_even = divisor_even_h * divisor_even_v;
    divisor_odd = divisor_odd_h * divisor_odd_v;
  } else {
    divisor_even = divisor_even_h * divisor_odd_v;
    divisor_odd = divisor_odd_h * divisor_even_v;
  }

  I_even_sum = I_even_sum / divisor_even;
  P_even_sum = P_even_sum / divisor_even;
  II_even_sum = II_even_sum / divisor_even;
  IP_even_sum = IP_even_sum / divisor_even;

  I_odd_sum = I_odd_sum / divisor_odd;
  P_odd_sum = P_odd_sum / divisor_odd;
  II_odd_sum = II_odd_sum / divisor_odd;
  IP_odd_sum = IP_odd_sum / divisor_odd;

  /////////////////////////////////////////////////////////// coefficient
  IP_even_sum = IP_even_sum - I_even_sum * P_even_sum;
  II_even_sum = II_even_sum - I_even_sum * I_even_sum;
  if (II_even_sum < THRESHOLD) II_even_sum = THRESHOLD;
  IP_even_sum = IP_even_sum / II_even_sum;

  IP_odd_sum = IP_odd_sum - I_odd_sum * P_odd_sum;
  II_odd_sum = II_odd_sum - I_odd_sum * I_odd_sum;
  if (II_odd_sum < THRESHOLD) II_odd_sum = THRESHOLD;
  IP_odd_sum = IP_odd_sum / II_odd_sum;

  bool rem = ((ow & 0x1) == 0);

  half red_coa_v = rem ? IP_odd_sum : IP_even_sum;
  red_coa[i] = red_coa_v;

  half red_cob_v = rem ? P_odd_sum - IP_odd_sum * I_odd_sum :
                          P_even_sum - IP_even_sum * I_even_sum;
  red_cob[i] = red_cob_v;

  half blue_coa_v = rem ? IP_even_sum : IP_odd_sum;
  blue_coa[i] = blue_coa_v;

  half blue_cob_v = rem ? P_even_sum - IP_even_sum * I_even_sum :
                           P_odd_sum - IP_odd_sum * I_odd_sum;
  blue_cob[i] = blue_cob_v;
}

__kernel void coefficient_vertical_sum(__global half2 *in, __global half2 *out, int H, int W) {
  int ow = get_global_id(0);
  W = W >> 1;

  if (ow >= W)
    return;

  half2 _v5, _v4, _v3, _v2, _v1, v, v1, v2, v3, v4, v5;

  //////////////////////////// first sum
  int oh = 0;
  half2 sum = 0;
  int i = ow;

  _v5 = _v4 = _v3 = _v2 = _v1 = 0;

  int idx = i;
  v = in[idx];
  sum += v;

  idx += W;
  v1 = in[idx];
  sum += v1;

  idx += W;
  v2 = in[idx];
  sum += v2;

  idx += W;
  v3 = in[idx];
  sum += v3;

  idx += W;
  v4 = in[idx];
  sum += v4;

  idx += W;
  v5 = in[idx];
  sum += v5;

  out[i] = sum;
  //////////////////////////////////  

  int ih = 5;
  for (int oh = 1; oh< H; oh++) {
    sum -= _v5;

    i += W;
    ih += 1;

    // update register (_v5~v5)
    _v5 = _v4;
    _v4 = _v3;
    _v3 = _v2;
    _v2 = _v1;
    _v1 = v;
    v = v1;
    v1 = v2;
    v2 = v3;
    v3 = v4;
    v4 = v5;  

    if (ih > H-1)
      v5 = 0;
    else
      v5 = in[ih*W+ow];

    sum += v5;

    out[i] = sum;
  }
}

__kernel void horizontal_sum_and_tentative_residual(__global half *green,
                                                    __global half *raw,
                                                    __global half4 *redA,
                                                    __global half4 *redB,
                                                    __global half4 *blueA,
                                                    __global half4 *blueB,
                                                    __global half *tentativeR,
                                                    __global half *tentativeB,
                                                    __global half *residualRB,
                                                    int H, int W) {
  int i = get_global_id(0);
  if (i >= H * W) return;
  int oh = i / W;
  int ow = i - oh*W;

  int quiotient = i >> 2;
  int remainder = i & 0x3;
  int iw;
  // vectors: pre_2, pre_1, mid, post_1, post_2
  // remainder == 0 -> sum = pre_2.w + pre_1.xyzw + mid.xyzw + post_1.xy
  // remainder == 1 -> sum =           pre_1.xyzw + mid.xyzw + post_1.xyz
  // remainder == 2 -> sum =           pre_1.yzw  + mid.xyzw + post_1.xyzw
  // remainder == 3 -> sum =           pre_1.zw   + mid.xyzw + post_1.xyzw + post_2.x
  // 

  int divisor_v, divisor_h;
  if( oh == 0 || oh == H-1 ) divisor_v = 6;
  else if( oh == 1 || oh == H-2 ) divisor_v = 7;
  else if( oh == 2 || oh == H-3 ) divisor_v = 8;
  else if( oh == 3 || oh == H-4 ) divisor_v = 9;
  else if( oh == 4 || oh == H-5 ) divisor_v = 10;
  else divisor_v = 11;

  //////////////////////////////////////////////////////////////// redA buffer
  half4 mid = redA[quiotient];
  half A_sum = mid.x + mid.y + mid.z + mid.w;

  /////////////////////////////////////////// pre_2 //
  half4 pre_2;
  if( ow >= 8 ) {
    pre_2 = redA[quiotient - 2];

    if (remainder == 0) {
      A_sum += pre_2.w;
    }
  }
  ////////////////////////////////////////////////////
  /////////////////////////////////////////// pre_1 //
  half4 pre_1;
  if (ow >= 4) {
    pre_1 = redA[quiotient - 1];
    A_sum += pre_1.z + pre_1.w;

    if (remainder < 3) {
      A_sum += pre_1.y;
    }
    if (remainder < 2) {
      A_sum += pre_1.x;
    }
  }
  ////////////////////////////////////////////////////
  ////////////////////////////////////////// post_1 //
  half4 post_1;
  if (ow < W-4) {
    post_1 = redA[quiotient + 1];
    A_sum += post_1.x + post_1.y;

    if (remainder > 0) {
      A_sum += post_1.z;
    }
    if (remainder > 1) {
      A_sum += post_1.w;
    }
  }
  ////////////////////////////////////////////////////
  ////////////////////////////////////////// post_2 //
  half4 post_2;
  if( ow < W-8 ) {
    post_2 = redA[quiotient + 2];

    if (remainder == 3) {
      A_sum += post_2.x;
    }
  }
  ////////////////////////////////////////////////////
  if (ow < 5)
    divisor_h = (6 + ow);
  else if (ow > W-6)
    divisor_h = (W + 5 - ow);
  else
    divisor_h = 11;
  divisor_h = divisor_h*divisor_v;
  //////////////////////////////////////////////////////////////// end redA

  //////////////////////////////////////////////////////////////// redB buffer
  mid = redB[quiotient];
  half B_sum = mid.x + mid.y + mid.z + mid.w;

  ////////////////////////////////////////////////////
  /////////////////////////////////////////// pre_2 //
  if( ow >= 8 ) {
    pre_2 = redB[quiotient - 2];
    if (remainder == 0)
      B_sum += pre_2.w;
  }
  ////////////////////////////////////////////////////
  /////////////////////////////////////////// pre_1 //
  if (ow >= 4) {
    pre_1 = redB[quiotient - 1];
    B_sum += pre_1.z + pre_1.w;
    if (remainder < 3)
      B_sum += pre_1.y;
    if (remainder < 2)
      B_sum += pre_1.x;
  }
  ////////////////////////////////////////////////////
  ////////////////////////////////////////// post_1 //
  if (ow < W-4) {
    post_1 = redB[quiotient + 1];
    B_sum += post_1.x + post_1.y;
    if (remainder > 0)
      B_sum += post_1.z;
    if (remainder > 1)
      B_sum += post_1.w;
  }
  ////////////////////////////////////////////////////
  ////////////////////////////////////////// post_2 //
  if( ow < W-8 ) {
    post_2 = redB[quiotient + 2];
    if (remainder == 3)
      B_sum += post_2.x;
  }
  //////////////////////////////////////////////////////////////// end redB

  //////////////////////////////////////////////////////////////// tentative red
  half value_green = green[i];
  half tentaR = (A_sum/divisor_h) * value_green + (B_sum/divisor_h);
  tentativeR[i] = tentaR;

  //////////////////////////////////////////////////////////////// blueA buffer
  mid = blueA[quiotient];
  A_sum = mid.x + mid.y + mid.z + mid.w;

  /////////////////////////////////////////// pre_2 //
  if( ow >= 8 ) {
    pre_2 = blueA[quiotient - 2];
  if (remainder == 0)
    A_sum += pre_2.w;
  }

  /////////////////////////////////////////// pre_1 //
  if (ow >= 4) {
    pre_1 = blueA[quiotient - 1];
    A_sum += pre_1.z + pre_1.w;
    if (remainder < 3)
      A_sum += pre_1.y;
    if (remainder < 2)
      A_sum += pre_1.x;
  }

  ////////////////////////////////////////// post_1 //
  if (ow < W-4) {
    post_1 = blueA[quiotient + 1];
    A_sum += post_1.x + post_1.y;
    if (remainder > 0)
      A_sum += post_1.z;
    if (remainder > 1)
      A_sum += post_1.w;
  }

  ////////////////////////////////////////// post_2 //
  if( ow < W-8 ) {
    post_2 = blueA[quiotient + 2];
    if (remainder == 3)
      A_sum += post_2.x;
  }
  //////////////////////////////////////////////////////////////// end blueA

  //////////////////////////////////////////////////////////////// blueB buffer
  mid = blueB[quiotient];
  B_sum = mid.x + mid.y + mid.z + mid.w;
  ////////////////////////////////////////////////////
  /////////////////////////////////////////// pre_2 //
  if( ow >= 8 ) {
    pre_2 = blueB[quiotient - 2];
  if (remainder == 0)
    B_sum += pre_2.w;
  }
  ////////////////////////////////////////////////////
  /////////////////////////////////////////// pre_1 //
  if (ow >= 4) {
    pre_1 = blueB[quiotient - 1];
    B_sum += pre_1.z + pre_1.w;
    if (remainder < 3)
      B_sum += pre_1.y;
    if (remainder < 2)
      B_sum += pre_1.x;
  }
  ////////////////////////////////////////////////////
  ////////////////////////////////////////// post_1 //
  if (ow < W-4) {
    post_1 = blueB[quiotient + 1];
    B_sum += post_1.x + post_1.y;
    if (remainder > 0)
      B_sum += post_1.z;
    if (remainder > 1)
      B_sum += post_1.w;
  }
  ////////////////////////////////////////////////////
  ////////////////////////////////////////// post_2 //
  if( ow < W-8 ) {
    post_2 = blueB[quiotient + 2];
    if (remainder == 3)
      B_sum += post_2.x;
  }
  ////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////// end blueB

  //////////////////////////////////////////////////////////////// tentative blue
  half tentaB = (A_sum/divisor_h) * value_green + (B_sum/divisor_h);
  tentativeB[i] = tentaB; 

  //////////////////////////////////////////////////////////////// residual red blue
  int oh_last_bit = oh & 0x1;
  int ow_last_bit = ow & 0x1;
  if( oh_last_bit == 0 && ow_last_bit == 1 ) {
    // red channel
    residualRB[i] = raw[i] - tentaR; 
  } else if( oh_last_bit == 1 && ow_last_bit == 0 ) {
    // blue channel
    residualRB[i] = raw[i] - tentaB; 
  }
}

