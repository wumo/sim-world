#ifndef BUF_LIBRARY_H
#define BUF_LIBRARY_H

#include <cstddef>
#include <algorithm>

void buf_init();

void maxOf(char *buf_a, char *buf_b, char *output, size_t size);

void concat(int axis, unsigned char **array, int size, int **shape, int shape_size,
            int byteSize, unsigned char *output);

void mulf(float *buf, size_t size, float s);

template<class A, class B>
void cast(A *a, B *b, size_t size);

#endif