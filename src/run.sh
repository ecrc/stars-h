cd .. &&
    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"/Users/mikhala/Applications/Conda/envs/py2_conmkl/lib"\
    && make clean all && testing/spatial.out
