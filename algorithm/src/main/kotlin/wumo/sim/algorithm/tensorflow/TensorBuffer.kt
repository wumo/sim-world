package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.ShortPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.newTensor
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.util.Dimension
import wumo.sim.util.*
import wumo.sim.util.ndarray.Buf
import wumo.sim.util.ndarray.NDArray
import java.nio.*

abstract class TensorBuffer<T> protected constructor(c_tensor: TF_Tensor) : Buf<T> {
  companion object {
    fun <T> toNDArray(tb: TensorBuffer<T>) = NDArray(Dimension(tb.dims), tb)
    fun <T> toNDArray(c_tensor: TF_Tensor) = toNDArray(invoke<T>(c_tensor))
    fun <T> fromNDArray(ndarray: NDArray<T>): TensorBuffer<T> {
      TODO()
    }
    
    operator fun <T> invoke(c_tensor: TF_Tensor): TensorBuffer<T> {
      val dtype = TF_TensorType(c_tensor)
      return when (dtype) {
        DT_COMPLEX64, DT_FLOAT -> FloatTensorBuffer(c_tensor)
        DT_DOUBLE -> DoubleTensorBuffer(c_tensor)
        DT_QINT32, DT_INT32 -> IntTensorBuffer(c_tensor)
        DT_BOOL -> BooleanTensorBuffer(c_tensor)
        DT_QUINT8, DT_UINT8, DT_QINT8, DT_INT8 -> ByteTensorBuffer(c_tensor)
        DT_BFLOAT16, DT_INT16 -> ShortTensorBuffer(c_tensor)
        DT_INT64 -> LongTensorBuffer(c_tensor)
        DT_STRING -> StringTensorBuffer(c_tensor)
        else -> throw IllegalStateException("invalid DataType($dtype)")
      } as TensorBuffer<T>
    }
    
    operator fun invoke(value: Float) = invoke(scalarDimension, f(value))
    operator fun invoke(value: Double) = invoke(scalarDimension, d(value))
    operator fun invoke(value: Boolean) = invoke(scalarDimension, B(value))
    operator fun invoke(value: Byte) = invoke(scalarDimension, b(value))
    operator fun invoke(value: Short) = invoke(scalarDimension, s(value))
    operator fun invoke(value: Int) = invoke(scalarDimension, i(value))
    operator fun invoke(value: Long) = invoke(scalarDimension, l(value))
    operator fun invoke(value: String) = invoke(scalarDimension, a(value))
    
    operator fun invoke(shape: Dimension, value: FloatArray) = FloatTensorBuffer(create(shape, FloatPointer(*value), DT_FLOAT))
    operator fun invoke(shape: Dimension, value: DoubleArray) = DoubleTensorBuffer(create(shape, DoublePointer(*value), DT_DOUBLE))
    operator fun invoke(shape: Dimension, value: BooleanArray) = BooleanTensorBuffer(create(shape, BytePointer(*ByteArray(value.size) { if (value[it]) 1 else 0 }), DT_BOOL))
    operator fun invoke(shape: Dimension, value: ByteArray) = ByteTensorBuffer(create(shape, BytePointer(*value), DT_INT8))
    operator fun invoke(shape: Dimension, value: ShortArray) = ShortTensorBuffer(create(shape, ShortPointer(*value), DT_INT16))
    operator fun invoke(shape: Dimension, value: IntArray) = IntTensorBuffer(create(shape, IntPointer(*value), DT_INT32))
    operator fun invoke(shape: Dimension, value: LongArray) = LongTensorBuffer(create(shape, LongPointer(*value), DT_INT64))
    operator fun invoke(shape: Dimension, array: Array<String>): StringTensorBuffer {
      val data = TFStringArray.encode(array)
      val t = newTensor(DT_STRING, shape.asLongArray(), data)
      return StringTensorBuffer(t, array)
    }
    
    private fun create(shape: Dimension, array: Pointer, dtype: Int): TF_Tensor {
      val total = array.sizeof() * array.limit()
      val byteSize = sizeof(dtype) * array.limit()
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
  }
}

class FloatTensorBuffer(c_tensor: TF_Tensor) : TensorBuffer<Float>(c_tensor) {
  private val buf = createBuffer<FloatBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Float) {
    buf.put(offset, data)
  }
}

class DoubleTensorBuffer(c_tensor: TF_Tensor) : TensorBuffer<Double>(c_tensor) {
  private val buf = createBuffer<DoubleBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Double) {
    buf.put(offset, data)
  }
}

class ByteTensorBuffer(c_tensor: TF_Tensor) : TensorBuffer<Byte>(c_tensor) {
  private val buf = createBuffer<ByteBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Byte) {
    buf.put(offset, data)
  }
}

class ShortTensorBuffer(c_tensor: TF_Tensor) : TensorBuffer<Short>(c_tensor) {
  private val buf = createBuffer<ShortBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Short) {
    buf.put(offset, data)
  }
}

class IntTensorBuffer(c_tensor: TF_Tensor) : TensorBuffer<Int>(c_tensor) {
  private val buf = createBuffer<IntBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Int) {
    buf.put(offset, data)
  }
}

class LongTensorBuffer(c_tensor: TF_Tensor) : TensorBuffer<Long>(c_tensor) {
  private val buf = createBuffer<LongBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Long) {
    buf.put(offset, data)
  }
}

class BooleanTensorBuffer(c_tensor: TF_Tensor) : TensorBuffer<Boolean>(c_tensor) {
  private val buf = createBuffer<ByteBuffer>()
  override fun get(offset: Int) = buf[offset].toInt() != 0
  
  override fun set(offset: Int, data: Boolean) {
    buf.put(offset, if (data) 1 else 0)
  }
}

class StringTensorBuffer(private var _c_tensor: TF_Tensor, val array: Array<String>? = null) : TensorBuffer<String>(_c_tensor) {
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
  
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: String) {
    buf[offset] = data
    modified = true
  }
}

object TFStringArray {
  val sizeofUInt64 = TensorBuffer.sizeof(DT_UINT64).toLong()
  
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