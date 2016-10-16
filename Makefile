# ------------------------
#  Usage:
#  	make [all]	-- make lib test
#  	make lib	-- make lib/libstars-h.a
#  	make shared	-- make lib/libstars-h.so (or libstars-h.dylib)
#  	make test	-- make test/*.out standalone examples
#  	make docs	-- make docs/html
#  	make clean	-- remove objects, libraries and executables

.PHONY:		all
all: 		lib test

# Loading Makefile vriables or setting them by default

-include make.inc

CC		?= cc
CFLAGS		?= -O2 -Wall -m64 -I${MKLROOT}/include
LDFLAGS		?=

ARCH		?= ar
ARCHFLAGS	?= rc
RANLIB		?= ranlib

INCLUDE		?= -I$(MKLROOT)/include
LIBS		?= -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib\
		   -lmkl_rt -lpthread -lm -ldl
STARSH_INCLUDE	= -Iinclude/
STARSH_LIB	= lib/libstarsh.a

# Actual Makefile content, building everything

STARSH_DIR	= src
STARSH_SRC	= $(wildcard $(STARSH_DIR)/*.c)
STARSH_OBJ	= $(STARSH_SRC:%.c=%.o)
STARSH_H	= $(wildcard include/stars*.h)
TEST_DIR	= testing
TEST_SRC	= $(wildcard $(TEST_DIR)/*.c)
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
	$(CC) $(CFLAGS) $(STARSH_INCLUDE) $(LIB_DIR) $< $(STARSH_LIB) $(LIBS) -o $@

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
