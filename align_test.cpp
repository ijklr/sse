#include <align.h>
#include <stdio.h>

int main()
{
    char *ptr = (char *)malloc_aligned(7, 100);

    printf("is 5 byte aligned = %s\n", (((size_t)ptr) % 5) ? "no" : "yes");
    printf("is 7 byte aligned = %s\n", (((size_t)ptr) % 7) ? "no" : "yes");

    free_aligned(ptr);

    return 0;
}
