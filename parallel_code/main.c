#include "demosaic.h"
#include "util.h"
#include <CL/cl.h>

const char fn_target_raw[] = "image/target.raw";
const char fn_output_png[] = "image/output.png";


int main()
{
	
	init_resources();

	read_rawfile(fn_target_raw);

	// Write to GPU from CPU
	write_gpu_buffer();

	// interpolation processing
	demosaic();

	// read from  GPU to CPU
	read_gpu_buffer();

	validation();

	save_png(fn_output_png);
	
	free_resources();

	return 0;
}


