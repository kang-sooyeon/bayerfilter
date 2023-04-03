#include "demosaic.h"
#include "util.h"


const char fn_target_raw[] = "image/target.raw";
const char fn_output_png[] = "image/output.png";
const char fn_target_buf[] = "image/target.buf";

int main()
{

	timer_start(0);
	init();
	printf("init() complete : %lf sec\n", timer_stop(0));

	timer_start(0);
	read_rawfile(fn_target_raw);
	printf("read_file() complete : %lf sec\n", timer_stop(0));

	timer_start(0);
	demosaic();
	printf("demosaic() complete : %lf sec\n", timer_stop(0));

	timer_start(0);
	//write_file(fn_target_buf);
	write_file_float("image/target_float2.buf");
	printf("save_png() complete : %lf\n", timer_stop(0));
	
	timer_start(0);
	save_png(fn_output_png);
	printf("save_png() complete : %lf\n", timer_stop(0));
	
	timer_start(0);
	free_resources();
	printf("free resources complete : %lf\n", timer_stop(0));

	return 0;
}


