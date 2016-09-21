make clean all
cd ../testing &&
    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"/Users/mikhala/Applications/Conda/envs/py2_conmkl/lib"\
    && make clean all && ./spatial.out
