"""
@copyright (c) 2017 King Abdullah University of Science and
                      Technology (KAUST). All rights reserved.

 STARS-H is a software package, provided by King Abdullah
             University of Science and Technology (KAUST)

 @file misc_scripts/code_generation/applications/particles/kernel_nd.py
 @version 0.1.0
 @author Aleksandr Mikhalev
 @date 2017-08-22
"""

from __future__ import print_function

import sys
import os

input_file = sys.argv[1]
output_folder = sys.argv[2]

with open(input_file, "r") as fd:
    lines = fd.readlines()

param_line = ""
for line in lines:
    ind = line.find("@generate")
    if ind != -1:
        param_line = line[ind+9:]
        break

split = param_line.split()
var_name = '@'+split[0]
var_values = split[2:]

output_file_base = os.path.join(output_folder, os.path.basename(input_file))
for val in var_values:
    output_file = "{}_{}d.c".format(output_file_base[:-2], val)
    print("{}".format(output_file), end=';')
    with open(output_file, "w") as fd:
        for line in lines:
            ind = line.find("@file")
            if ind != -1:
                line = "{} {}\n".format(line[:ind+5], output_file)
            ind = line.find("@generate")
            if ind != -1:
                line = " * Generated from file {} with NDIM={}\n".\
                        format(input_file, val, var_values)
            line = line.replace(var_name, val)
            fd.write(line)

