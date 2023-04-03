
TARGET=mainAA
OBJECTS=main.o util.o demosaic.o
CFLAGS=-std=c99 -O3 -Wall 

run:
	gcc -c demosaic.c $(CFLAGS)
	gcc -c util.c $(CFLAGS)
	gcc -o $(TARGET) main.c demosaic.o util.o $(CFLAGS)

	./mainAA

clean:
	rm -rf $(TARGET) $(OBJECTS)



