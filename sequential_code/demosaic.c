#include "demosaic.h"
#include "util.h"
#include <stdio.h>
#include "svpng.inc"
#include <stdlib.h>
#include <math.h>

#define C 3
size_t H = 3496;
size_t W = 4656;

size_t h = 5;
size_t v = 0;
size_t SIZE;

float* red;			// interpolated red : H * W
float* green;		// interpolated green : H * W
float* blue;		// interpolated blue : H * W
float* mosaic[C];	// mosaic data (one value for one channel) : H * W * channel
float* raw;			// raw CFA data (combine mosaic data) : H * W
float* mask[C];		// mask : H * W * channel
float* maskGb, *maskGr;

float* a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7;
float* b0, *b1, *b2, *b3, *b4, *b5, *b6, *b7;
float* c0, *c1, *c2, *c3, *c4, *c5, *c6, *c7;

float K11[11];

void init() {

	SIZE = H*W;
	set_ones(K11, 1, 11);

	// raw
	alloc_mat(&raw, H, W);

	// r, g, b
	alloc_mat(&green, H, W);
	alloc_mat(&red, H, W);
	alloc_mat(&blue, H, W);

	// mask, mosaic
	for( int i = 0; i < C; i++ ) {
		alloc_mat(&mask[i], H, W);
		alloc_mat(&mosaic[i], H, W);
	}

	// maksGb, maskGr
	alloc_mat(&maskGb, H, W);
	alloc_mat(&maskGr, H, W);

	// a, b, c variables	
	alloc_mat(&a0, H, W);
	alloc_mat(&a1, H, W);
	alloc_mat(&a2, H, W);
	alloc_mat(&a3, H, W);
	alloc_mat(&a4, H, W);
	alloc_mat(&a5, H, W);
	alloc_mat(&a6, H, W);
	alloc_mat(&a7, H, W);

	alloc_mat(&b0, H, W);
	alloc_mat(&b1, H, W);
	alloc_mat(&b2, H, W);
	alloc_mat(&b3, H, W);
	alloc_mat(&b4, H, W);
	alloc_mat(&b5, H, W);
	alloc_mat(&b6, H, W);
	alloc_mat(&b7, H, W);

	alloc_mat(&c0, H, W);
	alloc_mat(&c1, H, W);
	alloc_mat(&c2, H, W);
	alloc_mat(&c3, H, W);
	alloc_mat(&c4, H, W);
	alloc_mat(&c5, H, W);
	alloc_mat(&c6, H, W);
	alloc_mat(&c7, H, W);
}

void free_resources() {
	free(raw); free(green); free(red); free(blue); free(maskGr); free(maskGb); 
	for( int i = 0; i < C; i++ ) {
		free(mask[i]); free(mosaic[i]); 
	}
	free(a0); free(a1); free(a2); free(a3); free(a4); free(a5); free(a6); free(a7);
	free(b0); free(b1); free(b2); free(b3); free(b4); free(b5); free(b6); free(b7);
	free(c0); free(c1); free(c2); free(c3); free(c4); free(c5); free(c6); free(c7);
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


void read_rawfile(const char* filePath) {

	FILE* in = NULL;
	in = fopen(filePath, "rb");

	if (in == NULL) {
		printf("read_file open error\n");
		return;
	}
	short* buf = (short*)malloc(H * W * sizeof(short));
	fread(buf, sizeof(short), H * W, in);
	fclose(in);

	for (int i = 0; i < H*W; i++) {
		raw[i] = ((float)buf[i] / 1023.0) * 32;
	}
	free(buf);

}

void write_file(const char* filePath) {
	unsigned char *result = (unsigned char*)malloc(H*W*C*sizeof(unsigned char));
	for (int i = 0; i < H * W; i++ ) {
		result[i*3+0] = (unsigned char)(red[i] * 8);
		result[i*3+1] = (unsigned char)(green[i] * 8);
		result[i*3+2] = (unsigned char)(blue[i] * 8);
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
}

void write_file_float(const char* filePath, float* in) {
	float *result = (float*)malloc(H*W*C*sizeof(float));
	for (int i = 0; i < H * W; i++ ) {
		result[i*3+0] = red[i] * 8;
		result[i*3+1] = green[i] * 8;
		result[i*3+2] = blue[i] * 8;
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
}


void write_array(const char* filePath, float* in) {

	FILE* out = NULL;
	out = fopen(filePath, "wb");
	if (out == NULL) {
		printf("write_file error\n");
		return;
	}
	fwrite(in, sizeof(float), H * W, out);
	fclose(out);
}

void save_png(const char* filePath) {

	unsigned char *result = (unsigned char*)malloc(H*W*C*sizeof(unsigned char));
	for (int i = 0; i < H * W; i++ ) {
		result[i*3+0] = (unsigned char)(red[i] * 8);
		result[i*3+1] = (unsigned char)(green[i] * 8);
		result[i*3+2] = (unsigned char)(blue[i] * 8);
	}

	FILE* out = NULL;
	out = fopen(filePath, "wb");
	if (out == NULL) {
		printf("save_png error\n");
		return;
	}
	svpng(out, W, H, result, 0);
	fclose(out);
	free(result);
}

void set_bayer_mask() {

	for( int ih = 0; ih < H; ih++ ) {
		for( int iw = 0; iw < W; iw++ ) {
			int idx = ih * W + iw;
			// bayer format
			// G R
			// B G
			if( ih%2 == 0 ) {
				if( iw%2 == 0 ) {
					// G
					mask[1][idx] = 1;
					maskGr[idx] = 1;

					mask[0][idx] = 0;
					mask[2][idx] = 0;
					maskGb[idx] = 0;
				} else {
					// R
					mask[0][idx] = 1;

					mask[1][idx] = 0;
					mask[2][idx] = 0;
					maskGb[idx] = 0;
					maskGr[idx] = 0;

				}
			} else {
				if( iw%2 == 0 ) {
					// B
					mask[2][idx] = 1;

					mask[0][idx] = 0;
					mask[1][idx] = 0;
					maskGb[idx] = 0;
					maskGr[idx] = 0;
				} else {
					// G
					mask[1][idx] = 1;
					maskGb[idx] = 1;

					mask[0][idx] = 0;
					mask[2][idx] = 0;
					maskGr[idx] = 0;
				}
			}
		}
	}
	for (size_t c = 0; c < C; c++) {
		mul_mat(raw, mask[c], mosaic[c]);
	}

}


void demosaic() {

	//timer_start(1);
	set_bayer_mask();
	//printf("set mask complete : %lf sec\n", timer_stop(1));
	//timer_start(1);
	green_interpolation();
	//printf("green interpol complete : %lf sec\n", timer_stop(1));
	//timer_start(1);
	red_interpolation();
	//printf("red interpol complete : %lf sec\n", timer_stop(1));
	//timer_start(1);
	blue_interpolation();
	//printf("blue interpol complete : %lf sec\n", timer_stop(1));
}

void green_interpolation() {

	///////////////////////////////////////////////////////////// calculate horizontal and vertical color difference
	h = 5;
	v = 0;

	///////////////////////////////////////////////////////////// guide image
	float _Kh[] = { 1.0 / 2.0, 0., 1.0 / 2.0 };
	// b0 : rawh, b1 : rawv
	imfilter1D_h_replicate(raw, _Kh, 3, b0);
	imfilter1D_v_replicate(raw, _Kh, 3, b1);

	// b2 : a = mosaic[1] * maskGr
	mul_mat(mosaic[1], maskGr, b2);
	// b3 : b = mosaic[1] * maskGb
	mul_mat(mosaic[1], maskGb, b3);

	// b4 : Guiderh = mosaic[0] + rawh * maskGr;
	mul_mat(b0, maskGr, b4);
	add_mat(b4, mosaic[0], b4);	

	// a0 : tentativeGrh // guidedfilter(Guiderh, a, maskGr, h, v, eps, tentativeGrh);
	guidedfilter(b4, b2, maskGr, h, v, a0);

	// b4 : Guidebh = mosaic[2] + rawh * maskGb;
	mul_mat(b0, maskGb, b4);
	add_mat(b4, mosaic[2], b4);
	// a1 : tenttativeGbh // guidedfilter(Guidebh, b, maskGb, h, v, eps, tentativeGbh);
	guidedfilter(b4, b3, maskGb, h, v, a1);

	// b4 : Guidegh = mosaic[1] + rawh * mask[0] + rawh * mask[2];
	mul_mat(b0, mask[0], b4);
	mul_mat(b0, mask[2], b5);
	add_mat(b4, b5, b4);
	add_mat(b4, mosaic[1], b4);

	// a2 : tenttativeRh // guidedfilter(Guidegh, mosaic[0], mask[0], h, v, eps, tentativeRh);
	guidedfilter(b4, mosaic[0], mask[0], h, v, a2);

	// a3 : tenttativeBh // guidedfilter(Guidegh, mosaic[2], mask[2], h, v, eps, tentativeBh);
	guidedfilter(b4, mosaic[2], mask[2], h, v, a3);

	// b4 : Guiderv = mosaic[0] + rawv * maskGb;
	mul_mat(b1, maskGb, b4);
	add_mat(b4, mosaic[0], b4);

	// a4 : tenttativeGbv // guidedfilter(Guiderv, b, maskGb, v, h, eps, tentativeGrv);
	guidedfilter(b4, b3, maskGb, v, h, a4);

	// b4 : Guidebv = mosaic[2] + rawv * maskGr;
	mul_mat(b1, maskGr, b4);
	add_mat(b4, mosaic[2], b4);
	// a5 : tenttativeGrv // guidedfilter(Guidebv, a, maskGr, v, h, eps, tentativeGbv);
	guidedfilter(b4, b2, maskGr, v, h, a5);

	// b4 : Guidegv = mosaic[1] + rawv * mask[0] + rawv * mask[2];
	mul_mat(b1, mask[0], b4);
	mul_mat(b1, mask[2], b5);
	add_mat(b4, b5, b4);
	add_mat(b4, mosaic[1], b4);
	// a6 : tenttativeRv // guidedfilter(Guidegv, mosaic[0], mask[0], v, h, eps, tentativeRv);
	guidedfilter(b4, mosaic[0], mask[0], v, h, a6);

	// a7 : tenttativeBv // guidedfilter(Guidegv, mosaic[2], mask[2], v, h, eps, tentativeBv);
	guidedfilter(b4, mosaic[2], mask[2], v, h, a7);


	///////////////////////////////////////////////////////////// residual
	// b0 : residualGrh = mosaic[1] - tentativeGrh) * maskGr;
	sub_mat(mosaic[1], a0, b0);
	mul_mat(b0, maskGr, b0);

	// b1 : residualGbh = mosaic[1] - tentativeGbh) * maskGb;
	sub_mat(mosaic[1], a1, b1);
	mul_mat(b1, maskGb, b1);

	// b2 : residualRh = mosaic[0] - tentativeRh) * mask[0];
	sub_mat(mosaic[0], a2, b2);
	mul_mat(b2, mask[0], b2);

	// b3 : residualBh = mosaic[2] - tentativeBh) * mask[2];
	sub_mat(mosaic[2], a3, b3);
	mul_mat(b3, mask[2], b3);

	// b4 : residualGrv = mosaic[1] - tentativeGrv) * maskGb;
	sub_mat(mosaic[1], a4, b4);
	mul_mat(b4, maskGb, b4);

	// b5 : residualGbv = mosaic[1] - tentativeGbv) * maskGr;
	sub_mat(mosaic[1], a5, b5);
	mul_mat(b5, maskGr, b5);

	// b6 : residualRv = mosaic[0] - tentativeRv) * mask[0];
	sub_mat(mosaic[0], a6, b6);
	mul_mat(b6, mask[0], b6);

	// b7 : residualBv = mosaic[2] - tentativeBv) * mask[2];
	sub_mat(mosaic[2], a7, b7);
	mul_mat(b7, mask[2], b7);

	///////////////////////////////////////////////////////////// residual interpolation
	// c0-7 : residual
	imfilter1D_h(b0, _Kh, 3, c0);
	imfilter1D_h(b1, _Kh, 3, c1);
	imfilter1D_h(b2, _Kh, 3, c2);
	imfilter1D_h(b3, _Kh, 3, c3);
	imfilter1D_v(b4, _Kh, 3, c4);
	imfilter1D_v(b5, _Kh, 3, c5);
	imfilter1D_v(b6, _Kh, 3, c6);
	imfilter1D_v(b7, _Kh, 3, c7);

	// c0 : Grh = tentativeGrh + residualGrh) * mask[0];
	add_mat(c0, a0, c0);
	mul_mat(c0, mask[0], c0);

	// c1 : Gbh = tentativeGbh + residualGbh) * mask[2];
	add_mat(c1, a1, c1);
	mul_mat(c1, mask[2], c1);

	// c2 : Rh = tentativeRh + residualRh) * maskGr;
	add_mat(c2, a2, c2);
	mul_mat(c2, maskGr, c2);

	// c3 : Bh = tentativeBh + residualBh) * maskGb;
	add_mat(c3, a3, c3);
	mul_mat(c3, maskGb, c3);


	// c4 : Grv = tentativeGrv + residualGrv) * mask[0];
	add_mat(c4, a4, c4);
	mul_mat(c4, mask[0], c4);

	// c5 : Gbv = tentativeGbv + residualGbv) * mask[2];
	add_mat(c5, a5, c5);
	mul_mat(c5, mask[2], c5);

	// c6 : Rv = tentativeRv + residualRv) * maskGb;
	add_mat(c6, a6, c6);
	mul_mat(c6, maskGb, c6);

	// c7 : Bv = tentativeBv + residualBv) * maskGr;
	add_mat(c7, a7, c7);
	mul_mat(c7, maskGr, c7);

	///////////////////////////////////////////////////////////// vertical and horizontal color difference
	// c0 : difh = mosaic[1] + Grh + Gbh - mosaic[0] - mosaic[2] - Rh - Bh;
	add_mat(mosaic[1], c0, c0);
	add_mat(c0, c1, c0);
	sub_mat(c0, mosaic[0], c0);
	sub_mat(c0, mosaic[2], c0);
	sub_mat(c0, c2, c0);
	sub_mat(c0, c3, c0);


	// c4 : difv = mosaic[1] + Grv + Gbv - mosaic[0] - mosaic[2] - Rv - Bv;
	add_mat(mosaic[1], c4, c4);
	add_mat(c4, c5, c4);
	sub_mat(c4, mosaic[0], c4);
	sub_mat(c4, mosaic[2], c4);
	sub_mat(c4, c6, c4);
	sub_mat(c4, c7, c4);

	///////////////////////////////////////////////////////////// combine vertical and horizontal color differences
	// color difference gradient
	float _Kh2[] = { 1.0, 0.0, -1.0 };

	// a0 : difh2, a4_d : difv2
	imfilter1D_h_replicate(c0, _Kh2, 3, a0);
	imfilter1D_v_replicate(c4, _Kh2, 3, a4);
	abs_mat(a0);
	abs_mat(a4);

	// directional weight
	// b0 : wh, b4 : wv
	float K[25];
	set_ones(K, 5, 5);

	imfilter2D(a0, K, 5, 5, b0);
	imfilter2D(a4, K, 5, 5,  b4);

	// a0_d : Ww, a1_d : We, a2_d : Wn, a3_d : Ws
	float _Kw[5] = { 1., 0., 0., 0., 0. };
	imfilter1D_h_replicate(b0, _Kw, 5, a0);
	imfilter1D_v_replicate(b4, _Kw, 5, a2);

	float _Ke[5] = { 0., 0., 0., 0., 1. };
	imfilter1D_h_replicate(b0, _Ke, 5, a1);
	imfilter1D_v_replicate(b4, _Ke, 5, a3);

	// a4 : setones()
	set_ones(a4, H, W);
	// a0 : Ww = 1 ./ (Ww * Ww + exp(-32));
	mul_mat(a0, a0, a0);
	//add_val(a0, exp(-32.), a0);
	div_mat(a4, a0, a0);

	// a1 : We = 1 ./ (We * We + exp(-32));
	mul_mat(a1, a1, a1);
	//add_val(a1, exp(-32.), a1);
	div_mat(a4, a1, a1);

	// a2 : Wn = 1 ./ (Wn * Wn + exp(-32));
	mul_mat(a2, a2, a2);
	//add_val(a2, exp(-32.), a2);
	div_mat(a4, a2, a2);

	// a3 : Ws = 1 ./ (Ws * Ws + exp(-32));
	mul_mat(a3, a3, a3);
	//add_val(a3, exp(-32.), a3);
	div_mat(a4, a3, a3);


	///////////////////////////////////////////////////////////// combine directional color differences
	// c2 : Wt = Ww + We + Wn + Ws;
	add_mat(a0, a1, c2);
	add_mat(c2, a2, c2);
	add_mat(c2, a3, c2);

	
	// b2 : difn, b3_d : difs, b0_d : difw, b1_d : dife
	float _Kp[9] = { 0, 0, 0, 0, 0.570350, 0.345934, 0.077188, 0.006336, 0.000192 };		
	//// b1_d : dife, b3_d : difs
	imfilter1D_h_replicate(c0, _Kp, 9, b1);
	imfilter1D_v_replicate(c4, _Kp, 9, b3);

	float _Kq[9] = { 0.000192, 0.006336, 0.077188, 0.345934, 0.570350, 0, 0, 0, 0 };	
	//// b0_d : difw, b2_d : difn
	imfilter1D_h_replicate(c0, _Kq, 9, b0);
	imfilter1D_v_replicate(c4, _Kq, 9, b2);

	// c2 : dif = (Wn * difn + Ws * difs + Ww * difw + We * dife) / Wt;
	mul_mat(a2, b2, a2);
	mul_mat(a3, b3, a3);
	mul_mat(a0, b0, a0);
	mul_mat(a1, b1, a1);
	add_mat(a0, a1, a0);
	add_mat(a2, a3, a2);
	add_mat(a0, a2, a0);
	div_mat(a0, c2, c2);

	///////////////////////////////////////////////////////////// calcualte Green by adding bayer raw data
	// green = dif + raw;
	add_mat(c2, raw, green);

	// green = green * imask[1] + raw * mask[1];
	mul_mat_inv(green, mask[1], a0);
	mul_mat(raw, mask[1], a1);
	add_mat(a0, a1, green);

	// clip to 0-255
	clip(green, 0, 31);
}


void red_interpolation() {

	h = 5;
	v = 5;

	// a0 : tentativeR, guidedfilter(green, mosaic[0], mask[0], h, v, eps, tentativeR);
	guidedfilter(green, mosaic[0], mask[0], h, v, a0);

	// a1 : _residualR = (mosaic[0] - tentativeR) * mask[0];
	sub_mat(mosaic[0], a0, a1);
	mul_mat(a1, mask[0], a1);

	float _K[] = { 1.0/4.0, 1.0/2.0, 1.0/4.0, 1.0/2.0, 1.0, 1.0/2.0, 1.0/4.0, 1.0/2.0, 1.0/4.0 };

	// a2 : residualR
	imfilter2D(a1, _K, 3, 3, a2);
	
	//red = residualR + tentativeR;
	add_mat(a2, a0, red);
}

void blue_interpolation() {
	h = 5;
	v = 5;

	// a0 : tentativeB, guidedfilter(green, mosaic[2], mask[2], h, v, eps, tentativeR);
	guidedfilter(green, mosaic[2], mask[2], h, v, a0);

	// a1 : _residualB = (mosaic[2] - tentativeB) * mask[2];
	sub_mat(mosaic[2], a0, a1);
	mul_mat(a1, mask[2], a1);

	float _K[] = { 1.0/4.0, 1.0/2.0, 1.0/4.0, 1.0/2.0, 1.0, 1.0/2.0, 1.0/4.0, 1.0/2.0, 1.0/4.0 };
	// a2 : residualB
	imfilter2D(a1, _K, 3, 3, a2);

	// blue = residualB + tentativeB;
	add_mat(a2, a0, blue);
}



void imfilter2D(float* in, float* filter, int R, int S, float* out) {

	int padR = R / 2;
	int padS = S / 2;

	for (int oh = 0; oh < H; ++oh) {
		for (int ow = 0; ow < W; ++ow) {
			float x = 0;
			for (int r = 0; r < R; ++r) {
				for (int s = 0; s < S; ++s) {
					int ih = oh - padR + r;
					int iw = ow - padS + s;
					float ii;

					if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
					ii = in[ih * W + iw];

					float ff = filter[r * S + s];
					x += ii * ff;
				}
			}
			out[oh * W + ow] = x;
		}
	}
}

void imfilter2D_replicate(float* in, float* filter, int R, int S, float* out) {

	int padR = R / 2;
	int padS = S / 2;

	for (int oh = 0; oh < H; ++oh) {
		for (int ow = 0; ow < W; ++ow) {
			float x = 0;
			for (int r = 0; r < R; ++r) {
				for (int s = 0; s < S; ++s) {
					int ih = oh - padR + r;
					int iw = ow - padS + s;
					float ii;
					if (ih < 0 ) ih = 0;
					else if( ih >= H ) ih = H-1;

					if( iw < 0 ) iw = 0;
					else if( iw >= W ) iw = W-1;

					ii = in[ih * W + iw];

					float ff = filter[r * S + s];
					x += ii * ff;
				}
			}
			out[oh * W + ow] = x;
		}
	}
}

void imfilter1D_v_replicate(float* in, float* filter, int R, float* out) {

	int padR = R / 2;

	for (int oh = 0; oh < H; ++oh) {
		for (int ow = 0; ow < W; ++ow) {
			float x = 0;

			for (int pad = -padR; pad <= padR; ++pad) {
				float ii;
				int ih = oh + pad;

				if( ih < 0 ) ih = 0;
				else if( ih >= H ) ih = H-1;

				ii = in[ih * W + ow];

				float ff = filter[pad+padR];
				x += ii * ff;
			}
			out[oh * W + ow] = x;
		}
	}
}

void imfilter1D_v(float* in, float* filter, int R, float* out) {

	int padR = R / 2;

	for (int oh = 0; oh < H; ++oh) {
		for (int ow = 0; ow < W; ++ow) {
			float x = 0;
			for (int pad = -padR; pad <= padR; ++pad) {
				float ii;
				int ih = oh + pad;

				if( ih < 0 ) continue;
				else if( ih >= H ) continue;

				ii = in[ih * W + ow];

				float ff = filter[pad+padR];
				x += ii * ff;
			}
			out[oh * W + ow] = x;
		}
	}
}

void imfilter1D_h_replicate(float* in, float* filter, int R, float* out) {
	int padR = R / 2;

	for (int oh = 0; oh < H; ++oh) {
		for (int ow = 0; ow < W; ++ow) {
			float x = 0;
			for (int pad = -padR; pad <= padR; ++pad) {
				float ii;
				int iw = ow + pad;

				if( iw < 0 ) iw = 0;
				else if( iw >= W ) iw = W-1;

				ii = in[oh * W + iw];

				float ff = filter[pad+padR];
				x += ii * ff;
			}
			out[oh * W + ow] = x;
		}
	}
}

void imfilter1D_h(float* in, float* filter, int R, float* out) {

	int padR = R / 2;

	for (int oh = 0; oh < H; ++oh) {
		for (int ow = 0; ow < W; ++ow) {
			float x = 0;
			for (int pad = -padR; pad <= padR; ++pad) {
				float ii;
				int iw = ow + pad;

				if( iw < 0 ) continue;
				else if( iw >= W ) continue;

				ii = in[oh * W + iw];

				float ff = filter[pad+padR];
				x += ii * ff;
			}
			out[oh * W + ow] = x;
		}
	}
}



void guidedfilter(float* I, float* p, float* M, int h, int v, float* out) {

	// c0_d : mean_I, c1_d : mean_p, c2_d : mean_Ip, c3_d : mean_II
	// c4_d : cov_Ip, c5_d : N, c6_d : N2
	// b5_d : bf, b6_d : a, b7_d : b

	// threshold parameter
	float th = 0.00001 * 255 * 255;

	// c5 : N // boxfilter(mask_d, h, v, N);
	boxfilter(M, h, v, c5);	
	replace_new(c5, 0, 1);

	// b6 : a.set_ones();
	set_ones(b6, H, W);
	// c6 : N2 
	boxfilter(b6, h, v, c6);

	// c0 : mean_I = boxfilter(I.*M, h, v)./N;
	mul_mat(I, M, b6);
	boxfilter(b6, h, v, b5);
	div_mat(b5, c5, c0);

	// c1 : mean_p = boxfilter(p, h, v)./N;
	boxfilter(p, h, v, b5);
	div_mat(b5, c5, c1);

	// c2 : mean_Ip = boxfilter(I.*p, h, v)./N;
	mul_mat(I, p, b6);
	boxfilter(b6, h, v, b5);
	div_mat(b5, c5, c2);

	// c3 : mean_II = boxfilter(I.*I.*M, h, v)./N;
	mul_mat(I, I, b6);
	mul_mat(b6, M, b6);
	boxfilter(b6, h, v, b5);
	div_mat(b5, c5, c3);

	// the covariance of (I, p) in each local patch
	// c4 : cov_Ip = mean_Ip - mean_I * mean_p;
	mul_mat(c0, c1, c4);
	sub_mat(c2, c4, c4);

	// b6 : var_I = mean_II - mean_I.*mean_I;
	mul_mat(c0, c0, b6);
	sub_mat(c3, b6, b6);
	replace_th(b6, th);	// less than theshold, replace th

	// linear coefficients
	// b6 : a = cov_Ip./(var_I);
	div_mat(c4, b6, b6);
	// b7 : b = mean_p - a.*mean_I;
	mul_mat(b6, c0, b7);
	sub_mat(c1, b7, b7);

	// c0 : mean_a = boxfilter(a, h, v)./N2;
	boxfilter(b6, h, v, b5);
	div_mat(b5, c6, c0);

	// c1 : mean_b = boxfilter(b, h, v)./N2;
	boxfilter(b7, h, v, b5);
	div_mat(b5, c6, c1);

	// output
	// out = mean_a .* I + mean_b;
	mul_mat(c0, I, out);
	add_mat(out, c1, out);
}

void boxfilter(float* in, int h, int v, float *out) {

	float* cum = c7;
	if( h==5 && v==0 )
		imfilter1D_h(in, K11, 11, out);
	else if( h==0 && v ==5 ){
		imfilter1D_v(in, K11, 11, out);}
	else if( h==5 && v ==5 ) {
		imfilter1D_v(in, K11, 11, c7);
		imfilter1D_h(c7, K11, 11, out);
	}
}
