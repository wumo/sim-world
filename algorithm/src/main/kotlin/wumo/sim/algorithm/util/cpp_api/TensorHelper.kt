package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.tensorflow.*
import org.bytedeco.javacpp.tensorflow.Tensor.create
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.tuples.tuple2
import java.nio.*

fun tensor(vararg value: Float) =
    create(value, MakeShape(value.size))

fun tensor(vararg value: Double) =
    create(value, MakeShape(value.size))

fun tensor(vararg value: Byte) =
    create(value, MakeShape(value.size))

fun tensor(vararg value: Short) =
    create(value, MakeShape(value.size))

fun tensor(vararg value: Int) =
    create(value, MakeShape(value.size))

fun tensor(vararg value: Long) =
    create(value, MakeShape(value.size))

fun tensor(vararg value: String) =
    create(value, MakeShape(value.size))

fun tensor(shape: Dimension, vararg data: Float) =
    create(data, MakeShape(shape.asLongArray()))

fun tensor(shape: Dimension, vararg data: Double) =
    create(data, MakeShape(shape.asLongArray()))

fun tensor(shape: Dimension, vararg data: Byte) =
    create(data, MakeShape(shape.asLongArray()))

fun tensor(shape: Dimension, vararg data: Short) =
    create(data, MakeShape(shape.asLongArray()))

fun tensor(shape: Dimension, vararg data: Int) =
    create(data, MakeShape(shape.asLongArray()))

fun tensor(shape: Dimension, vararg data: Long) =
    create(data, MakeShape(shape.asLongArray()))

fun tensor(shape: Dimension, vararg data: String) =
    create(data, MakeShape(shape.asLongArray()))

fun <T> tensor(other: WrappedTensor<T>): Tensor {
  return other.nativeTensor
}

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

abstract class WrappedTensor<T>(internal val nativeTensor: Tensor) : Iterable<T> {
  protected val stride: IntArray
  protected val dims: IntArray
  protected val numElements: Int
  
  init {
    val shape = nativeTensor.shape()
    stride = IntArray(shape.dims())
    dims = IntArray(shape.dims()) { shape.dim_size(it).toInt() }
    var n = 0
    if (dims.isNotEmpty()) {
      stride[stride.lastIndex] = 1
      for (a in stride.lastIndex - 1 downTo 0)
        stride[a] = dims[a + 1] * stride[a + 1]
      n = dims[0] * stride[0]
    }
    numElements = n
  }
  
  abstract fun buf_get(offset: Int): T
  abstract fun buf_set(offset: Int, data: T)
  
  private inline fun <U> get_set(vararg idx: Int, op: (Int) -> U): U {
    require(idx.size == dims.size)
    var offset = 0
    for ((i, value) in idx.withIndex()) {
      require(value in 0 until dims[i]) { "dim($i)=$value is out of range[0,${dims[i]})" }
      offset += value * stride[i]
    }
    return op(offset)
  }
  
  operator fun get(vararg idx: Int) = get_set(*idx) { buf_get(it) }
  
  operator fun set(vararg idx: Int, data: T) = get_set(*idx) { buf_set(it, data) }
  
  override fun iterator() = object : Iterator<T> {
    var a = 0
    override fun hasNext() = a < numElements
    
    override fun next() = buf_get(a++)
  }
  
  fun withIndex() = object : Iterator<tuple2<IntArray, T>> {
    val idx = IntArray(dims.size).apply { this[this.lastIndex] = -1 }
    var a = 0
    lateinit var element: tuple2<IntArray, T>
    override fun hasNext() = a < numElements
    
    override fun next(): tuple2<IntArray, T> {
      for (_idx in dims.lastIndex downTo 0) {
        idx[_idx]++
        if (idx[_idx] < dims[_idx])
          break
        idx[_idx] = 0
      }
      val value = buf_get(a++)
      if (this::element.isInitialized)
        element._2 = value
      else
        element = tuple2(idx, value)
      return element
    }
    
  }
}

class FloatTensor(nativeTensor: Tensor) : WrappedTensor<Float>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<FloatBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
  override fun buf_set(offset: Int, data: Float) {
    buf.put(offset, data)
  }
}

class DoubleTensor(nativeTensor: Tensor) : WrappedTensor<Double>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<DoubleBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
  override fun buf_set(offset: Int, data: Double) {
    buf.put(offset, data)
  }
}

class IntTensor(nativeTensor: Tensor) : WrappedTensor<Int>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<IntBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
  override fun buf_set(offset: Int, data: Int) {
    buf.put(offset, data)
  }
}

class LongTensor(nativeTensor: Tensor) : WrappedTensor<Long>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<LongBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
  override fun buf_set(offset: Int, data: Long) {
    buf.put(offset, data)
  }
}

class ByteTensor(nativeTensor: Tensor) : WrappedTensor<Byte>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<ByteBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
  override fun buf_set(offset: Int, data: Byte) {
    buf.put(offset, data)
  }
}

class ShortTensor(nativeTensor: Tensor) : WrappedTensor<Short>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<ShortBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset]
  override fun buf_set(offset: Int, data: Short) {
    buf.put(offset, data)
  }
}

class BooleanTensor(nativeTensor: Tensor) : WrappedTensor<Boolean>(nativeTensor) {
  private val buf = nativeTensor.createBuffer<ByteBuffer>()
  
  override fun buf_get(offset: Int) = buf[offset].toInt() != 0
  override fun buf_set(offset: Int, data: Boolean) {
    buf.put(offset, if (data) 1 else 0)
  }
}

class StringTensor(nativeTensor: Tensor) : WrappedTensor<String>(nativeTensor) {
  
  private val buf = nativeTensor.createStringArray()
//      .let { sa ->
//    Array(nativeTensor.NumElements().toInt()) {
//      sa.position(it.toLong()).toString()
//    }
//  }
  
  override fun buf_get(offset: Int) = buf.position(offset.toLong()).data().string
  override fun buf_set(offset: Int, data: String) {
    buf.position(offset.toLong()).put(data)
  }
}