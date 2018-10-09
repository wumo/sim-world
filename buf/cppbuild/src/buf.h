#ifndef BUF_LIBRARY_H
#define BUF_LIBRARY_H

#include <cstddef>
#include <algorithm>

void buf_init();

void maxOf(char *buf_a, char *buf_b, char *output, size_t size);

void concat(int axis, unsigned char **array, int **shape, size_t size,
            size_t byteSize, unsigned char *output, int *result_shape);

void divf(float *buf, size_t size, float s);

template<class A, class B>
void cast(A *a, B *b, size_t size);

#endif