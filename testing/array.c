#include <stdio.h>
#include <stdlib.h>
#include "stars.h"

int main(int argc, char **argv)
{
    int i;
    int ndim = 4;
    int shape[4] = {2, 3, 4, 5};
    STARS_Array *array = STARS_Array_new(4, shape, 'd');
    double *buffer = array->buffer;
    for(i = 0; i < array->size; i++)
    {
        buffer[i] = i;
    }
    STARS_Array_info(array);
    STARS_Array_print(array);
    //STARS_Array_trans(array);
    STARS_Array_tomatrix(array, 'C');
    STARS_Array_info(array);
    STARS_Array_print(array);
    STARS_Array_free(array);
}
