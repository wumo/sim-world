package wumo.sim.util.ndarray

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.IntPointer
import org.bytedeco.javacpp.Pointer.memcpy
import org.bytedeco.javacpp.PointerPointer
import wumo.sim.buf
import wumo.sim.util.native

fun <T : Any> concat(array: List<NDArray<T>>, axis: Int = 0): NDArray<T> {
  val size = array.size
  val first = array[0]
  val dtype = first.dtype
  val byteSize = dtype.byteSize
  val dataPtr = PointerPointer<BytePointer>(size.toLong())
  val shapePtr = PointerPointer<IntPointer>(size.toLong())
  var sumAlongDim = 0
  val resultShape = array[0].shape.copy()
  for (i in 0 until size) {
    val nd = array[i]
    shapePtr.put(i.toLong(), IntPointer(*nd.shape.asIntArray()!!))
    dataPtr.put(i.toLong(), nd.native)
    sumAlongDim += nd.shape[axis]
  }
  resultShape[axis] = sumAlongDim
  val resultPtr = BytePointer(resultShape.numElements() * byteSize.toLong())
  buf.concat(axis, dataPtr, size, shapePtr, first.shape.rank, byteSize, resultPtr)
  return NDArray(resultShape, BytePointerBuf(resultPtr, dtype))
}

fun <T : Any> concatenate(array: List<NDArray<T>>, axis: Int = 0): NDArray<T> {
  val size = array.size
  val first = array[0]
  val byteSize = first.dtype.byteSize
  val resultShape = first.shape.copy()
  val preInclude = IntArray(size)
  for (i in 0 until size) {
    val dim = array[i].shape[axis]
    preInclude[i] += dim + if (i > 0) preInclude[i - 1] else 0
  }
  resultShape[axis] = preInclude.last()
  val resultPtr = BytePointer((resultShape.numElements() * first.dtype.byteSize).toLong())
  val dimStride = first.stride[axis]
  val globalStride = (dimStride * preInclude.last() * byteSize).toLong()
  for (i in 0 until size) {
    val nd = array[i]
    val dim = nd.shape[axis]
    var offsetDst = ((preInclude[i] - dim) * dimStride * byteSize).toLong()
    val range = dim * dimStride
    val localStride = (range * byteSize).toLong()
    var offsetSrc = 0L
    for (k in 0 until nd.shape.numElements() / range) {
      val dst = resultPtr.position(offsetDst)
      val src = nd.native.position(offsetSrc)
      memcpy(dst, src, localStride)
      offsetDst += globalStride
      offsetSrc += localStride
    }
    nd.native.position(0)
  }
  resultPtr.position(0)
  return NDArray(resultShape, BytePointerBuf(resultPtr, first.dtype))
}

fun <T : Any> concatenateParallel(array: List<NDArray<T>>, axis: Int = 0): NDArray<T> {
  val size = array.size
  val first = array[0]
  val byteSize = first.dtype.byteSize
  val resultShape = first.shape.copy()
  val preInclude = IntArray(size)
  for (i in 0 until size) {
    val dim = array[i].shape[axis]
    preInclude[i] += dim + if (i > 0) preInclude[i - 1] else 0
  }
  resultShape[axis] = preInclude.last()
  val resultPtr = BytePointer((resultShape.numElements() * first.dtype.byteSize).toLong())
  val dimStride = first.stride[axis]
  val globalStride = (dimStride * preInclude.last() * byteSize).toLong()
  (0 until size).toList().parallelStream().forEach { i ->
    native {
      val resultPtr = BytePointer(resultPtr)
      val nd = array[i]
      val dim = nd.shape[axis]
      var offsetDst = ((preInclude[i] - dim) * dimStride * byteSize).toLong()
      val range = dim * dimStride
      val localStride = (range * byteSize).toLong()
      var offsetSrc = 0L
      for (k in 0 until nd.shape.numElements() / range) {
        val dst = resultPtr.position(offsetDst)
        val src = nd.native.position(offsetSrc)
        memcpy(dst, src, localStride)
        offsetDst += globalStride
        offsetSrc += localStride
      }
      nd.native.position(0)
    }
  }
  return NDArray(resultShape, BytePointerBuf(resultPtr, first.dtype))
}
