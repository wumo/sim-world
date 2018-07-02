package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.tensorflow.*
import org.bytedeco.javacpp.tensorflow.Tensor.create
import wumo.sim.algorithm.util.Dimension
import java.nio.*

fun TF_CPP.tensor(shape: Dimension, vararg data: Float) =
    create(data, MakeShape(shape.asLongArray()))

fun TF_CPP.tensor(vararg value: Float) =
    create(value, MakeShape(value.size))

object TensorHelper {
  inline fun <reified T> wrap(nativeTensor: Tensor): WrappedTensor<T> {
    val dtype = nativeTensor.dtype()
    if (T::class.java != Any::class.java) {
      val expected = dtypeFromClass(T::class.java)
      require(expected == dtype) { "expected $expected(${T::class.java}) but acutal is $dtype" }
    }
    return when (dtype) {
      DT_BOOL -> BooleanTensor(nativeTensor)
      DT_FLOAT -> FloatTensor(nativeTensor)
      DT_DOUBLE -> DoubleTensor(nativeTensor)
      DT_INT8 -> ByteTensor(nativeTensor)
      DT_INT16 -> ShortTensor(nativeTensor)
      DT_INT32 -> IntTensor(nativeTensor)
      DT_INT64 -> LongTensor(nativeTensor)
      DT_STRING -> StringTensor(nativeTensor)
      else -> throw IllegalArgumentException("Not supported: $dtype")
    } as WrappedTensor<T>
  }
}

abstract class WrappedTensor<T>(nativeTensor: Tensor) {
  protected val stride: IntArray
  protected val dims: IntArray
  
  init {
    val shape = nativeTensor.shape()
    stride = IntArray(shape.dims())
    dims = IntArray(shape.dims()) { shape.dim_size(it).toInt() }
    if (dims.isNotEmpty()) {
      stride[stride.lastIndex] = 1
      for (a in stride.lastIndex - 1 downTo 0)
        stride[a] = dims[a + 1] * stride[a + 1]
    }
  }
  
  abstract fun buf_get(offset: Int): T
  
  operator fun get(vararg idx: Int): T {
    require(idx.size == dims.size)
    var offset = 0
    for ((i, value) in idx.withIndex()) {
      require(value in 0 until dims[i]) { "dim($i)=$value is out of range[0,${dims[i]})" }
      offset += value * stride[i]
    }
    return buf_get(offset)
  }
}

class FloatTensor(nativeTensor: Tensor) : WrappedTensor<Float>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<FloatBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
}

class DoubleTensor(nativeTensor: Tensor) : WrappedTensor<Double>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<DoubleBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
}

class IntTensor(nativeTensor: Tensor) : WrappedTensor<Int>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<IntBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
}

class LongTensor(nativeTensor: Tensor) : WrappedTensor<Long>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<LongBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
}

class ByteTensor(nativeTensor: Tensor) : WrappedTensor<Byte>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<ByteBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
}

class ShortTensor(nativeTensor: Tensor) : WrappedTensor<Short>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<ShortBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
}

class BooleanTensor(nativeTensor: Tensor) : WrappedTensor<Boolean>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<ByteBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset].toInt() != 0
}

class StringTensor(nativeTensor: Tensor) : WrappedTensor<String>(nativeTensor) {
  private val buf = nativeTensor.createStringArray().let { sa ->
    Array(nativeTensor.NumElements().toInt()) {
      sa.position(it.toLong()).toString()
    }
  }
  
  override fun buf_get(offset: Int) = buf[offset]
}