#ifndef INCLUDE_ALIGN_H
#define INCLUDE_ALIGN_H
#include <stddef.h>
void *malloc_aligned(size_t alignment, size_t bytes);
void free_aligned(void *raw_data);
#endif
