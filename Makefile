# ------------------------
#  Usage:
#  	make [all]	-- make lib test
#  	make lib	-- make lib/libstars-h.a
#  	make shared	-- make lib/libstars-h.so (or libstars-h.dylib)
#  	make test	-- make testing/*.out standalone examples
#  	make docs	-- make docs/html
#  	make clean	-- remove objects, libraries and executables

.PHONY:		all
all: 		lib test

# Loading Makefile vriables or setting them by default

-include make.inc

CC		?= gcc
CFLAGS		?= -O3 -Wall -m64 -I${MKLROOT}/include -std=c11 -fopenmp
#LDFLAGS		?=-L/Users/mikhala/Downloads/lapack-3.6.1 -L/Users/mikhala/Applications/HPC/lib/
CFLAGS		+= $(shell pkg-config --cflags starpu-1.2)
LDFLAGS		+= $(shell pkg-config --libs starpu-1.2)

ARCH		?= ar
ARCHFLAGS	?= rc
RANLIB		?= ranlib

INCLUDE		?= -I$(MKLROOT)/include
LIBS		?= -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib\
		   -lmkl_rt -liomp5 -lm
#LIBS		?= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a\
		   ${MKLROOT}/lib/intel64/libmkl_sequential.a\
		   ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group\
		   -lpthread -lm -ldl
#LIBS		?=  ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl
STARSH_INCLUDE	= -Iinclude/
STARSH_LIB	= lib/libstarsh.a

# Actual Makefile content, building everything

STARSH_DIR	= src
#STARSH_SRC	= $(wildcard $(STARSH_DIR)/*.c)
CONTROL_SRC	= $(wildcard $(STARSH_DIR)/control/*.c)
BACKEND_SRC	= $(wildcard $(STARSH_DIR)/backends/sequential/*.c)
MISC_SRC	= $(STARSH_DIR)/misc.c
APPS_SRC	= $(STARSH_DIR)/applications/spatial.c
STARSH_SRC	= $(CONTROL_SRC) $(BACKEND_SRC) $(MISC_SRC) $(APPS_SRC)
STARSH_OBJ	= $(STARSH_SRC:%.c=%.o)
STARSH_H	= $(wildcard include/stars*.h)
TEST_DIR	= testing
#TEST_SRC	= $(wildcard $(TEST_DIR)/*.c)
TEST_SRC	= $(TEST_DIR)/sequential.c
TEST_OBJ	= $(TEST_SRC:%.c=%.o)
TEST_EXE	= $(TEST_SRC:%.c=%.out)

# Build libraries

.PHONY:		lib lib_dir

lib:		lib_dir $(STARSH_LIB)

lib_dir:
	mkdir -p lib

$(STARSH_LIB):	$(STARSH_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(STARSH_OBJ)
	$(RANLIB) $@

# Build executables (tests and examples)

.PHONY:		test

test:		$(TEST_EXE)

%.out:		%.c $(STARSH_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(STARSH_INCLUDE) $(LIB_DIR) $< $(STARSH_LIB) $(LIBS) -o $@

# Cleaning everything

.PHONY:		clean

clean:
	-rm -f $(STARSH_OBJ) $(STARSH_LIB) $(TEST_OBJ) $(TEST_EXE)

# Debugging

.PHONY:		echo

echo:
	@echo "CC          $(CC)"
	@echo "CFLAGS      $(CFLAGS)"
	@echo
	@echo "STARSH_DIR  $(STARSH_DIR)"
	@echo "STARSH_SRC  $(STARSH_SRC)"
	@echo "STARSH_OBJ  $(STARSH_OBJ)"
	@echo "STARSH_H    $(STARSH_H)"
	@echo
	@echo "TEST_DIR    $(TEST_DIR)"
	@echo "TEST_SRC    $(TEST_SRC)"
	@echo "TEST_EXE    $(TEST_EXE)"

# Compiling every required object file from a C-source

%.o:		%.c $(STARSH_H)
	$(CC) $(CFLAGS) $(INCLUDE) $(STARSH_INCLUDE) -c $< -o $@
