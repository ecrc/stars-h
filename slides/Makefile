.PHONY:		all clean

SRC	= $(wildcard *.tex)
PDF	= $(SRC:%.tex=%.pdf)

all:		$(PDF) 

%.pdf:		%.tex
	mkdir -p build
	pdflatex --shell-escape -output-directory=build $<
	mv build/$@ $@

clean:
	rm -rf build/
	rm -f *.pdf
