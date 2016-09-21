CC	= gcc
CFLAGS	= -O2
LIBS	= -lmkl_rt
STARS	= ../stars-h.a
INCLUDE	= ../include

all:	spatial.out

.c.out:
	$(CC) $(CFLAGS) -I $(INCLUDE) $< $(STARS) -o $@ $(LIBS) -L/Users/mikhala/Applications/Conda/envs/py2_conmkl/lib

clean:
	rm -rf *.out
