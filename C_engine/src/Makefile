# _toto_Render/C_engine/src/Makefile
CC = gcc
CFLAGS = -fPIC -O2 -Wall -fopenmp  # Added -fopenmp here
LIBFLAGS = -shared
OUTFILE = libanalysis_engine.so

OBJS = analysis_engine.o

all: $(OUTFILE)

analysis_engine.o: analysis_engine.c analysis_engine.h
	$(CC) $(CFLAGS) -c analysis_engine.c -o analysis_engine.o

$(OUTFILE): analysis_engine.o
	$(CC) $(LIBFLAGS) $(CFLAGS) -o $(OUTFILE) analysis_engine.o  # Added $(CFLAGS) here to include -fopenmp

clean:
	rm -f *.o *.so
	