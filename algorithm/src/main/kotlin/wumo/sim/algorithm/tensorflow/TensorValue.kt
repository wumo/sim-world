package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.Pointer.memcpy
import org.bytedeco.javacpp.ShortPointer
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.tuples.tuple2
import java.lang.reflect.Array
import java.nio.Buffer
import java.nio.FloatBuffer


abstract class TensorValue<T>(val c_tensor: TF_Tensor) : Iterable<T> {
  companion object {
    fun <T> wrap(c_tensor: TF_Tensor): TensorValue<T> {
      val dtype = TF_TensorType(c_tensor)
      return when (dtype) {
        DT_COMPLEX64, DT_FLOAT -> FloatTensorValue(c_tensor)
        else -> throw IllegalStateException("invalid DataType($dtype)")
      } as TensorValue<T>
    }
    
    fun create(shape: Dimension, value: FloatArray) =
        FloatTensorValue(create(shape, FloatPointer(*value), DT_FLOAT))
    
    fun create(shape: Dimension, value: DoubleArray) =
        FloatTensorValue(create(shape, DoublePointer(*value), DT_FLOAT))
    
    fun create(shape: Dimension, value: ByteArray) =
        FloatTensorValue(create(shape, BytePointer(*value), DT_FLOAT))
    
    fun create(shape: Dimension, value: ShortArray) =
        FloatTensorValue(create(shape, ShortPointer(*value), DT_FLOAT))
    
    fun create(shape: Dimension, value: IntArray) =
        FloatTensorValue(create(shape, IntPointer(*value), DT_FLOAT))
    
    fun create(shape: Dimension, value: LongArray) =
        FloatTensorValue(create(shape, LongPointer(*value), DT_FLOAT))
    
    fun create(shape: Dimension, value: StringArray) =
        FloatTensorValue(create(shape, FloatPointer(*value), DT_FLOAT))
    
    private fun create(shape: Dimension, array: Pointer, dtype: Int): TF_Tensor {
      val numElements = Array.getLength(array)
      assert(shape.numElements().toInt() == numElements)
      val byteSize = (sizeof(dtype) * numElements).toLong()
      val num_dims = shape.rank().toInt()
      val c_tensor = TF_AllocateTensor(dtype, shape.asLongArray(), num_dims, byteSize)
      val data = TF_TensorData(c_tensor)
      memcpy(data, array, byteSize)
    }
    
    private fun sizeof(dtype: Int): Int {
      return when (dtype) {
        DT_BOOL, DT_QUINT8, DT_UINT8, DT_QINT8, DT_INT8 -> 1
        DT_BFLOAT16, DT_INT16 -> 2
        DT_COMPLEX64, DT_FLOAT, DT_QINT32, DT_INT32 -> 4
        DT_DOUBLE, DT_INT64 -> 8
        else -> throw IllegalArgumentException("$dtype tensors do not have a fixed element size")
      }
    }
  }
  
  protected val stride: IntArray
  protected val dims: IntArray
  protected val numElements: Int
  val dtype = TF_TensorType(c_tensor)
  val numDims = TF_NumDims(c_tensor)
  
  init {
    stride = IntArray(numDims)
    dims = IntArray(numDims) { TF_Dim(c_tensor, it).toInt() }
    var n = 0
    if (dims.isNotEmpty()) {
      stride[stride.lastIndex] = 1
      for (a in stride.lastIndex - 1 downTo 0)
        stride[a] = dims[a + 1] * stride[a + 1]
      n = dims[0] * stride[0]
    }
    numElements = n
  }
  
  
  @Suppress("IMPLICIT_CAST_TO_ANY")
  protected fun <B : Buffer> createBuffer(): B {
    val ptr = BytePointer(TF_TensorData(c_tensor))
    val size = TF_TensorByteSize(c_tensor)
    return when (dtype) {
      DT_COMPLEX64, DT_FLOAT -> FloatPointer(ptr).position(0L).capacity((size / 4)).asBuffer()
      DT_DOUBLE -> DoublePointer(ptr).position(0L).capacity((size / 8)).asBuffer()
      DT_QINT32, DT_INT32 -> IntPointer(ptr).position(0L).capacity((size / 4)).asBuffer()
      DT_BOOL, DT_QUINT8, DT_UINT8, DT_QINT8, DT_INT8 -> ptr.position(0L).capacity(size).asBuffer()
      DT_BFLOAT16, DT_INT16 -> ShortPointer(ptr).position(0L).capacity((size / 2)).asBuffer()
      DT_INT64 -> LongPointer(ptr).position(0L).capacity((size / 8)).asBuffer()
      else -> throw IllegalStateException("invalid DataType($dtype)")
    } as B
//    byteBuf.capacity(sz)
//    return byteBuf.asByteBuffer().order(ByteOrder.nativeOrder())
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
  
  @Suppress("NAME_SHADOWING")
  private fun printTensor(dim: Int, offset: Int, sb: StringBuilder) {
    val padding = StringBuilder().let { s ->
      repeat(dim + 1) { s.append(' ') }
      s.toString()
    }
    var offset = offset
    val isVector = dim == dims.lastIndex - 1
    sb.append('[')
    for (i in 0 until dims[dim]) {
      if (i != 0) sb.append(padding)
      if (isVector)
        printVector(offset, stride[dim], sb)
      else
        printTensor(dim + 1, offset, sb)
      if (i != dims[dim] - 1)
        sb.append(",\n")
      offset += stride[dim]
    }
    sb.append(']')
  }
  
  private fun printVector(offset: Int, size: Int, sb: StringBuilder) {
    sb.append('[')
    for (j in 0 until size) {
      sb.append(buf_get(offset + j))
      if (j != size - 1)
        sb.append(',')
    }
    sb.append(']')
  }
  
  override fun toString(): String {
    val sb = StringBuilder()
    when (dims.size) {
      0 -> return get().toString()
      1 -> printVector(0, dims[0], sb)
      else -> printTensor(0, 0, sb)
    }
    return sb.toString()
  }
}

class FloatTensorValue(c_tensor: TF_Tensor) : TensorValue<Float>(c_tensor) {
  private val buf = createBuffer<FloatBuffer>()
  override fun buf_get(offset: Int) = buf[offset]
  
  override fun buf_set(offset: Int, data: Float) {
    buf.put(offset, data)
  }
}