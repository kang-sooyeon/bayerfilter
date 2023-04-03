
TARGET=main
OBJECTS=main.o demosaic.o
CFLAGS=-std=c99 -O3 -Wall -lm -lOpenCL -I /usr/local/cuda/include

run:
	gcc -c demosaic.c $(CFLAGS)
	gcc -o $(TARGET) main.c demosaic.o $(CFLAGS)

	./main	
clean:
	rm main
	rm $(OBJECTS)
	



