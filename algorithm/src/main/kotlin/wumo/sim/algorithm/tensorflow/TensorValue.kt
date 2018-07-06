package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.ShortPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.newTensor
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.tuples.tuple2
import java.nio.*


abstract class TensorValue<T>(c_tensor: TF_Tensor) : Iterable<T> {
  companion object {
    fun <T> wrap(c_tensor: TF_Tensor): TensorValue<T> {
      val dtype = TF_TensorType(c_tensor)
      return when (dtype) {
        DT_COMPLEX64, DT_FLOAT -> FloatTensorValue(c_tensor)
        DT_DOUBLE -> DoubleTensorValue(c_tensor)
        DT_QINT32, DT_INT32 -> IntTensorValue(c_tensor)
        DT_BOOL -> BooleanTensorValue(c_tensor)
        DT_QUINT8, DT_UINT8, DT_QINT8, DT_INT8 -> ByteTensorValue(c_tensor)
        DT_BFLOAT16, DT_INT16 -> ShortTensorValue(c_tensor)
        DT_INT64 -> LongTensorValue(c_tensor)
        DT_STRING -> StringTensorValue(c_tensor)
        else -> throw IllegalStateException("invalid DataType($dtype)")
      } as TensorValue<T>
    }
    
    fun create(shape: Dimension, value: FloatArray) =
        FloatTensorValue(create(shape, FloatPointer(*value), DT_FLOAT))
    
    fun create(shape: Dimension, value: DoubleArray) =
        DoubleTensorValue(create(shape, DoublePointer(*value), DT_DOUBLE))
    
    fun create(shape: Dimension, value: ByteArray) =
        ByteTensorValue(create(shape, BytePointer(*value), DT_INT8))
    
    fun create(shape: Dimension, value: ShortArray) =
        ShortTensorValue(create(shape, ShortPointer(*value), DT_INT16))
    
    fun create(shape: Dimension, value: IntArray) =
        IntTensorValue(create(shape, IntPointer(*value), DT_INT32))
    
    fun create(shape: Dimension, value: LongArray) =
        LongTensorValue(create(shape, LongPointer(*value), DT_INT64))
    
    fun create(shape: Dimension, array: Array<String>): StringTensorValue {
      assert(shape.numElements() == array.size.toLong())
      val data = TFStringArray.encode(array)
      val t = newTensor(DT_STRING, shape.asLongArray(), data)
      return StringTensorValue(t, array)
    }
    
    private fun create(shape: Dimension, array: Pointer, dtype: Int): TF_Tensor {
      assert(shape.numElements() == array.limit())
      val byteSize = sizeof(dtype) * array.limit()
//      val c_tensor = allocateTensor(dtype, shape.asLongArray(), byteSize)
//      val data = TF_TensorData(c_tensor)
//      memcpy(data, array, byteSize)
//      return c_tensor
      return newTensor(dtype, shape.asLongArray(), BytePointer(array).capacity(byteSize))
    }
    
    internal fun sizeof(dtype: Int): Int {
      return when (dtype) {
        DT_BOOL, DT_QUINT8, DT_UINT8, DT_QINT8, DT_INT8 -> 1
        DT_BFLOAT16, DT_INT16 -> 2
        DT_COMPLEX64, DT_FLOAT, DT_QINT32, DT_INT32 -> 4
        DT_DOUBLE, DT_INT64, DT_UINT64 -> 8
        else -> throw IllegalArgumentException("$dtype tensors do not have a fixed element size")
      }
    }
  }
  
  open val c_tensor = c_tensor
  protected val stride: LongArray
  protected val dims: LongArray
  protected val numElements: Long
  val dtype = TF_TensorType(c_tensor)
  val numDims = TF_NumDims(c_tensor)
  
  init {
    stride = LongArray(numDims)
    dims = LongArray(numDims) { TF_Dim(c_tensor, it) }
    var n = 0L
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
    var offset = 0L
    for ((i, value) in idx.withIndex()) {
      require(value in 0 until dims[i]) { "dim($i)=$value is out of range[0,${dims[i]})" }
      offset += value * stride[i]
    }
    return op(offset.toInt())
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
      if (i != 0L) sb.append(padding)
      if (isVector)
        printVector(offset, stride[dim].toInt(), sb)
      else
        printTensor(dim + 1, offset, sb)
      if (i != dims[dim] - 1)
        sb.append(",\n")
      offset += stride[dim].toInt()
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
      1 -> printVector(0, dims[0].toInt(), sb)
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

class DoubleTensorValue(c_tensor: TF_Tensor) : TensorValue<Double>(c_tensor) {
  private val buf = createBuffer<DoubleBuffer>()
  override fun buf_get(offset: Int) = buf[offset]
  
  override fun buf_set(offset: Int, data: Double) {
    buf.put(offset, data)
  }
}

class ByteTensorValue(c_tensor: TF_Tensor) : TensorValue<Byte>(c_tensor) {
  private val buf = createBuffer<ByteBuffer>()
  override fun buf_get(offset: Int) = buf[offset]
  
  override fun buf_set(offset: Int, data: Byte) {
    buf.put(offset, data)
  }
}

class ShortTensorValue(c_tensor: TF_Tensor) : TensorValue<Short>(c_tensor) {
  private val buf = createBuffer<ShortBuffer>()
  override fun buf_get(offset: Int) = buf[offset]
  
  override fun buf_set(offset: Int, data: Short) {
    buf.put(offset, data)
  }
}

class IntTensorValue(c_tensor: TF_Tensor) : TensorValue<Int>(c_tensor) {
  private val buf = createBuffer<IntBuffer>()
  override fun buf_get(offset: Int) = buf[offset]
  
  override fun buf_set(offset: Int, data: Int) {
    buf.put(offset, data)
  }
}

class LongTensorValue(c_tensor: TF_Tensor) : TensorValue<Long>(c_tensor) {
  private val buf = createBuffer<LongBuffer>()
  override fun buf_get(offset: Int) = buf[offset]
  
  override fun buf_set(offset: Int, data: Long) {
    buf.put(offset, data)
  }
}

class BooleanTensorValue(c_tensor: TF_Tensor) : TensorValue<Boolean>(c_tensor) {
  private val buf = createBuffer<ByteBuffer>()
  override fun buf_get(offset: Int) = buf[offset].toInt() != 0
  
  override fun buf_set(offset: Int, data: Boolean) {
    buf.put(offset, if (data) 1 else 0)
  }
}

class StringTensorValue(private var _c_tensor: TF_Tensor, val array: Array<String>? = null) : TensorValue<String>(_c_tensor) {
  private val buf = array ?: TFStringArray.decode(_c_tensor, numElements)
  var modified = false
  override val c_tensor
    get() =
      if (!modified)
        _c_tensor
      else {
        _c_tensor = newTensor(DT_STRING, dims, TFStringArray.encode(buf))
        modified = false
        _c_tensor
      }
  
  override fun buf_get(offset: Int) = buf[offset]
  
  override fun buf_set(offset: Int, data: String) {
    buf[offset] = data
    modified = true
  }
}

object TFStringArray {
  val sizeofUInt64 = TensorValue.sizeof(DT_UINT64).toLong()
  
  fun encode(array: Array<String>): BytePointer {
    // Compute bytes needed for encoding.
    var size = 0L
    for (s in array) {
      // uint64 starting_offset, TF_StringEncode-d string.
      size += sizeofUInt64 + TF_StringEncodedSize(s.length.toLong())
    }
    
    // Encode all strings.
    val base = BytePointer(size)
    val offsetBuf = LongPointer(base)
    val data_start = sizeofUInt64 * array.size
    var dst = data_start
    var dst_len = size - data_start
    val status = newStatus()
    var offsets = 0L
    for (s in array) {
      offsetBuf.position(offsets).put(dst - data_start)
      offsets++
      val consumed = TF_StringEncode(s, s.length.toLong(), base.position(dst), dst_len, status)
      throwExceptionIfNotOk(status)
      dst += consumed
      dst_len -= consumed
    }
    if (dst != size)
      throw IllegalArgumentException("invalid string tensor encoding (decoded $dst " +
                                     "bytes, but the tensor is encoded in $size bytes")
    return base.position(0L)
  }
  
  fun decode(src: TF_Tensor, numElements: Long): Array<String> {
    val input = BytePointer(TF_TensorData(src))
    val offsetBuf = LongPointer(input)
    val src_size = TF_TensorByteSize(src)
    if (src_size / sizeofUInt64 < numElements)
      throw IllegalArgumentException("Malformed TF_STRING tensor; too short to hold number of elements")
    val data_start = sizeofUInt64 * numElements
    val limit = src_size
    val status = newStatus()
    return Array(numElements.toInt()) {
      
      val offset = offsetBuf.get(it.toLong())
      if (offset >= limit - data_start)
        throw  IllegalArgumentException("Malformed TF_STRING tensor; element $it out of range")
      val len = SizeTPointer(1L)
      val p = BytePointer(1L)
      val srcp = data_start + offset
      TF_StringDecode(input.position(srcp), limit - srcp, p, len, status)
      throwExceptionIfNotOk(status)
      p.limit(len.get()).string
    }
  }
}