#include <stdlib.h>
#include <stdio.h>
#include "hpartitioning.h"

void *aux_get(void *data, int count, int *index)
{
    double *mydata = data;
    int i, j;
    double *aux = (double *)malloc(4*sizeof(double));
    j = index[0];
    aux[0] = mydata[2*j];
    aux[1] = aux[0];
    aux[2] = mydata[2*j+1];
    aux[3] = aux[2];
    for(i = 1; i < count; i++)
    {
        j = index[i];
        if(mydata[2*j] < aux[0])
            aux[0] = mydata[2*j];
        if(mydata[2*j] > aux[1])
            aux[1] = mydata[2*j];
        if(mydata[2*j+1] < aux[2])
            aux[2] = mydata[2*j+1];
        if(mydata[2*j+1] > aux[3])
            aux[3] = mydata[2*j+1];
    }
    return aux;
}

uint64_t *divide_cluster(void *data, int count, int *index)
{
    double *mydata = data;
    double *new_data = (double *)malloc(2*count*sizeof(double));
    uint64_t i, j;
    double mean[2] = {0., 0.};
    for(i = 0; i < count; i++)
    {
        new_data[2*i] = mydata[2*index[i]];
        new_data[2*i+1] = mydata[2*index[i]+1];
        mean[0] += new_data[2*i];
        mean[1] += new_data[2*i+1];
    }
    mean[0] = mean[0]/count;
    LAPACKE_dgesvd(2, );
}


int main(int argc, char **argv)
{
}
