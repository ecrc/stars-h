CC	= gcc
CFLAGS	= -O2
ARCH	= ar -cr
RANLIB	= ranlib
OBJ	= problem.o blr.o spatial.o misc.o
LIBNAME	= stars-h.a
INCLUDE = ../include

all:	../$(LIBNAME)

../$(LIBNAME):	$(OBJ)
	$(ARCH) $@ $^
	$(RANLIB) $@
.c.o:
	$(CC) $(CFLAGS) -I $(INCLUDE) -c $< -o $@

clean:
	rm -rf *.o
	rm -f ../$(LIBNAME)
