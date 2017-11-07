#!/usr/bin/env python

# This script updates following properties of STARS-H project: file name in
# each @file doxygen comment, version in each @version doxygen comment, date in
# each @date doxygen comment.

from runpy import run_path

run_path("update_filenames_in_docs.py")
run_path("update_version_and_date_in_docs.py")
