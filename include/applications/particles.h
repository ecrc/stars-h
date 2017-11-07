static int radix_sort(uint32_t *data, STARSH_int count, int ndim,
        STARSH_int *order);
static void radix_sort_recursive(uint32_t *data, STARSH_int count, int ndim,
        STARSH_int *order, STARSH_int *tmp_order, int sdim, int sbit,
        STARSH_int lo, STARSH_int hi);
