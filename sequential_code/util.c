#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "util.h"

static double start_time[8];

static double get_time() {
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

void alloc_mat(float **in, int H, int W) {
	  *in = (float *) malloc(sizeof(float) * H * W);
	  if (*in == NULL) {
	  	printf("Failed to allocate memory for matrix.\n");
		exit(0);
	  }
}


void set_zeros(float *in, int H, int W) {
	  memset(in, 0, sizeof(float) * H * W);
}

void set_ones(float *in, int H, int W) {
	//memset(in, 1, sizeof(float) * H * W);
	for( int i = 0; i < H*W; i++ ) {
		in[i] = 1;
	}
}

void abs_mat(float* in) {
	for( int i = 0; i < SIZE; i++ ) {
		if( in[i] < 0 ) in[i] = -in[i];
	}
}

void add_mat(float *A, float *B, float* C) {
	for( int i = 0; i < SIZE; i++ ) {
		C[i] = A[i] + B[i];
	}
}

void add_val(float *A, float v, float* C) {
	for( int i = 0; i < SIZE; i++ ) {
		C[i] = A[i] + v;
	}
}

void sub_mat(float *A, float *B, float* C) {
	for( int i = 0; i < SIZE; i++ ) {
		C[i] = A[i] - B[i];
	}
}

void mul_mat(float *A, float *B, float* C) {
	for( int i = 0; i < SIZE; i++ ) {
		C[i] = A[i] * B[i];
	}
}

void mul_mat_inv(float *A, float *B, float* C) {
	for( int i = 0; i < SIZE; i++ ) {
		C[i] = A[i] * !B[i];
	}
}

void div_mat(float *A, float *B, float* C) {
	for( int i = 0; i < SIZE; i++ ) {
		C[i] = A[i] / B[i];
	}
}

float max(float *in, int n) {
	float max_val = in[0];
	for (int i = 1; i < SIZE; i++) {
		if (max_val < in[i]) {
			max_val = in[i];
		}
	}
	return max_val;
}

void clip(float* in, int start, int end) {
	for (int i = 0; i < SIZE; i++) {
		float x = in[i];
		if( x < start ) in[i] = start;
		else if (x > end) in[i] = end;
	}
}

void cumsum(float* in, float* out, int axis) {

	int idx;
	float v;
	if (axis == 1) {
		// cumulative sum over Y axis
		for (int i = 0; i < W; i++) {
			v = 0;
			for (int j = 0; j < H; j++) {
				idx = j * W + i;
				out[idx] = in[idx] + v;
				v = out[idx];
			}
		}
	} else if (axis == 2) {
		// cumulative sum over X axis
		for (int i = 0; i < H; i++) {
			v = 0;
			for (int j = 0; j < W; j++) {
				idx = i * W + j;
				out[idx] = in[idx] + v;
				v = out[idx];
			}
		}
	}

}

void diff(float *in, float* out, int axis) {

	if( axis == 1 ) {
		// diff over Y axis
		for( int ih = 0; ih < H; ih++ ) {
			for( int iw = 0; iw < W; iw++ ) {
				if( ih >= 6 && ih <= H-6 ) out[ih*W+iw] = in[(ih+5)*W+iw] - in[(ih-6)*W+iw];
				else if( ih >= H-5 && ih <= H-1 ) out[ih*W+iw] = in[(H-1)*W+iw] - in[(ih-6)*W+iw];
				else if( ih >= 0 && ih <= 5 ) out[ih*W+iw] = in[(ih+5)*W+iw];
			}
		}

	} else if( axis == 2 ) {
		// diff over X axis	
		for( int iw = 0; iw < W; iw++ ) {
			for( int ih = 0; ih < H; ih++ ) {
				if( iw >= 6 && iw <= W-6 ) out[ih*W+iw] = in[ih*W+(iw+5)] - in[ih*W+(iw-6)];
				if( iw >= W-5 && iw <= W-1 ) out[ih*W+iw] = in[ih*W+(W-1)] - in[ih*W+(iw-6)];
				else if( iw >= 0 && iw <= 5 ) out[ih*W+iw] = in[ih*W+(iw+5)];
			}
		}
	}
}

void replace_new(float* in, float _old, float _new) {
	for (int i = 0; i < SIZE; i++) {
		if ( in[i] == _old ) {
			in[i] = _new;
		}
	}
}

void replace_th(float* in, float th) {
	for (int i = 0; i < SIZE; i++) {
		if ( in[i] < th ) {
			in[i] = th;
		}
	}
}

