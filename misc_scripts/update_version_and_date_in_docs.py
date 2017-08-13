#!/usr/bin/env python

# This script looks into every .c file in src/ directory every .h file in
# include/ directory and updates version and date in doxygen comment with
# @version and @date

from __future__ import print_function
from glob import glob
from datetime import date

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

with open("../VERSION.txt", "r") as fd:
    version = fd.readline()[:-1]

date = date.today()
strdate = str(date)

for fname in all_files:
    with open(fname, "r+") as fd:
        lines = fd.readlines()
        fd.seek(0)
        fd.truncate()
        for line in lines:
            ind = line.find(r"@version")
            if ind != -1:
                newline = line[:ind+9]+version+"\n"
                if newline != line:
                    print("Warning: updated version of {}".format(fname))
                line = newline
            ind = line.find(r"@date")
            if ind != -1:
                newline = line[:ind+6]+strdate+"\n"
                if newline != line:
                    print("Warning: updated date of {}".format(fname))
                line = newline
            fd.write(line)
    print("File {} was succesfully processed".format(fname))
