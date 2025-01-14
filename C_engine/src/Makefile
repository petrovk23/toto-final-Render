CC = gcc
CFLAGS = -fPIC -O3 -Wall -mavx2 -mfma -fopenmp -march=native
LIBFLAGS = -shared
OUTFILE = libanalysis_engine.so

OBJS = analysis_engine.o

all: $(OUTFILE)

analysis_engine.o: analysis_engine.c analysis_engine.h
	$(CC) $(CFLAGS) -c analysis_engine.c -o analysis_engine.o

$(OUTFILE): analysis_engine.o
	$(CC) $(CFLAGS) $(LIBFLAGS) -o $(OUTFILE) analysis_engine.o

clean:
	rm -f *.o *.so
