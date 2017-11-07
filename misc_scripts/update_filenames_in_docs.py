#!/usr/bin/env python

# This script looks into every .c file in src/ directory every .h file in
# include/ directory and updates filename in doxygen comment with @file

from __future__ import print_function
from glob import glob

src_files = glob("../src/*.c")
src_files.extend(glob("../src/*/*.c"))
src_files.extend(glob("../src/*/*/*.c"))
src_files.extend(glob("../src/*/*/*/*.c"))
src_files.extend(glob("../testing/*.c"))
src_files.extend(glob("../testing/*/*.c"))
src_files.extend(glob("../examples/*.c"))
src_files.extend(glob("../examples/*/*.c"))

h_files = glob("../include/*.h")

all_files = src_files+h_files

for fname in all_files:
    with open(fname, "r+") as fd:
        lines = fd.readlines()
        fd.seek(0)
        fd.truncate()
        for line in lines:
            ind = line.find(r"@file")
            if ind != -1:
                newline = line[:ind+6]+fname[3:]+"\n"
                if newline != line:
                    print("Warning: wrong file name was fixed in {}".
                            format(fname))
                line = newline
            fd.write(line)
    print("File {} was succesfully processed".format(fname))
