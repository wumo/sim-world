#include <algorithm>
#include "buf.h"

#define MAX_Of(x, y) ((x) >= (y)) ? (x) : (y)
#define CAST(A, B) template void cast< A , B >(A *a, B *b, size_t size);

void buf_init() {
}

void maxOf(char *buf_a, char *buf_b, char *output, size_t size) {
  for (size_t i = 0; i < size; ++i)
    output[i] = MAX_Of(buf_a[i], buf_b[i]);
}

void concat(const int axis, unsigned char **array, int **shape, size_t size, size_t byteSize,
            unsigned char *output, int *result_shape) {
  int *preInclude = new int[size]();
  for (int i = 0; i < size; ++i) {
    int dim = shape[i][axis];
    preInclude[i] += dim + ((i > 0) ? preInclude[i - 1] : 0);
  }
  
  delete[] preInclude;
}

void divf(float *buf, size_t size, float s) {
  for (int i = 0; i < size; ++i)
    buf[i] /= s;
}

template<class A, class B>
void cast(A *a, B *b, size_t size) {
  std::copy(a, a + size, b);
}

//char
CAST(unsigned char, unsigned char)

CAST(char, unsigned char)

CAST(short, unsigned char)

CAST(int, unsigned char)

CAST(long long, unsigned char)

CAST(float, unsigned char)

CAST(double, unsigned char)

//char
CAST(unsigned char, char)

CAST(char, char)

CAST(short, char)

CAST(int, char)

CAST(long long, char)

CAST(float, char)

CAST(double, char)
//short
CAST(unsigned char, short)

CAST(char, short)

CAST(short, short)

CAST(int, short)

CAST(long long, short)

CAST(float, short)

CAST(double, short)
//int
CAST(unsigned char, int)

CAST(char, int)

CAST(short, int)

CAST(int, int)

CAST(long long, int)

CAST(float, int)

CAST(double, int)
//long
CAST(unsigned char, long)

CAST(char, long)

CAST(short, long)

CAST(int, long)

CAST(long long, long)

CAST(float, long)

CAST(double, long)
//float
CAST(unsigned char, float)

CAST(char, float)

CAST(short, float)

CAST(int, float)

CAST(long long, float)

CAST(float, float)

CAST(double, float)
//double
CAST(unsigned char, double)

CAST(char, double)

CAST(short, double)

CAST(int, double)

CAST(long long, double)

CAST(float, double)

CAST(double, double)