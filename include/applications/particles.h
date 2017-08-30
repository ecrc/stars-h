static int radix_sort(uint32_t *data, size_t count, int ndim, size_t *order);
static void radix_sort_recursive(uint32_t *data, size_t count, int ndim,
        size_t *order, size_t *tmp_order, int sdim, int sbit,
        size_t lo, size_t hi);
