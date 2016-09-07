CC=clang
INCLUDE=../include/
LIBDIRS=/Users/mikhala/Applications/Conda/envs/py2_conmkl/lib/
LIBS=-lmkl_rt
WARNINGS=-Wall -Wno-incompatible-pointer-types

all:
	$(CC) -I$(INCLUDE) -L$(LIBDIRS) block_lowrank.c misc.c run.c $(LIBS) -o a.out $(WARNINGS)
