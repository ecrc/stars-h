all: lib testing

testing:	lib
	cd testing && make all

lib:	stars-h.a

stars-h.a:
	cd src && make all

clean:	cleanlib cleantest

cleanlib:
	cd src && make clean

cleantest:
	cd testing && make clean
