#include <algorithm>
#include <cstring>
#include "buf.h"
#include <eigen3/Eigen/Core>

#define MAX_Of(x, y) ((x) >= (y)) ? (x) : (y)
#define CAST(A, B) template void cast< A , B >(A *a, B *b, size_t size);

void buf_init() {
}

void maxOf(char *buf_a, char *buf_b, char *output, size_t size) {
//  auto a = Eigen::Map<Eigen::Array<char, 1, Eigen::Dynamic>>(buf_a, size);
//  auto b = Eigen::Map<Eigen::Array<char, 1, Eigen::Dynamic>>(buf_b, size);
//  auto c = a.cwiseMax(b);
//  Eigen::Map<Eigen::Array<char, 1, Eigen::Dynamic>>(output, size) = c;
  for (size_t i = 0; i < size; ++i)
    output[i] = MAX_Of(buf_a[i], buf_b[i]);
}

void concat(const int axis, unsigned char **array, int size,
            int **shape, int shape_size, int byteSize,
            unsigned char *output) {
  int *preInclude = new int[size]();
  for (int i = 0; i < size; ++i) {
    int dim = shape[i][axis];
    preInclude[i] += dim + ((i > 0) ? preInclude[i - 1] : 0);
  }
  
  int dimStride = 1;
  for (int i = shape_size - 1; i > axis; --i)
    dimStride *= shape[0][i];
  int numRange = 1;
  for (int i = 0; i < axis; ++i)
    numRange *= shape[0][i];
  int globalStride = dimStride * preInclude[size - 1] * byteSize;
  for (int i = 0; i < size; ++i) {
    unsigned char *nd = array[i];
    int dim = shape[i][axis];
    int offsetDst = (preInclude[i] - dim) * dimStride * byteSize;
    int localStride = dim * dimStride * byteSize;
    int offsetSrc = 0;
    for (int k = 0; k < numRange; ++k) {
      unsigned char *start = nd + offsetSrc;
      memcpy(output + offsetDst, start, localStride);
//      std::copy(start, start + localStride, output + offsetDst);
      offsetDst += globalStride;
      offsetSrc += localStride;
    }
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