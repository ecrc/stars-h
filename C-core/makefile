CC=clang
INCLUDE=../include/
LIBDIRS=/Users/mikhala/Applications/Conda/envs/py2_conmkl/lib/
LIBS=-lmkl_rt
WARNINGS=-Wall -Wno-incompatible-pointer-types
SRC=run.c blr.c misc.c kernel.c
all:
	$(CC) -I$(INCLUDE) -L$(LIBDIRS) $(LIBS) $(SRC) -o a.out $(WARNINGS)
