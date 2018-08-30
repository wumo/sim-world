@file:Suppress("UNCHECKED_CAST")

package wumo.sim.tensorflow.tensor

import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.Pointer.memcpy
import org.bytedeco.javacpp.ShortPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.allocateTensor
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.newTensor
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.core.check
import wumo.sim.tensorflow.dtypeToClass
import wumo.sim.tensorflow.types.*
import wumo.sim.util.*
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.Buf
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.implementation.*
import java.nio.*

abstract class Tensor<T : Any> protected constructor(c_tensor: TF_Tensor) : Buf<T> {
  companion object {
    private val convert_switch = SwitchType2<Shape, Tensor<*>>().apply {
      case<FloatArrayBuf> { Tensor(_2, _1.raw) }
      case<DoubleArrayBuf> { Tensor(_2, _1.raw) }
      case<BooleanArrayBuf> { Tensor(_2, _1.raw) }
      case<ByteArrayBuf> { Tensor(_2, _1.raw) }
      case<ShortArrayBuf> { Tensor(_2, _1.raw) }
      case<IntArrayBuf> { Tensor(_2, _1.raw) }
      case<LongArrayBuf> { Tensor(_2, _1.raw) }
      case<ArrayBuf<*>> { Tensor(_2, Array(_1.raw.size) { _1[it].toString() }) }
    }
    
    fun <T : Any> toNDArray(tb: Tensor<T>) = NDArray(Shape(tb.dims), tb, dtypeToClass(tb.dtype.baseDataType.cValue))
    fun <T : Any> toNDArray(c_tensor: TF_Tensor) = toNDArray(invoke<T>(c_tensor))
    fun <T : Any> fromNDArray(ndarray: NDArray<T>) = (if (ndarray.raw is Tensor<*>) ndarray.raw
    else convert_switch(ndarray.raw, ndarray.shape)) as Tensor<T>
    
    private val create_switch = SwitchValue<Int, TF_Tensor, Tensor<*>>().apply {
      case(DT_COMPLEX64, DT_FLOAT) { FloatTensor(it) }
      case(DT_DOUBLE) { DoubleTensor(it) }
      case(DT_QINT32, DT_INT32) { IntTensor(it) }
      case(DT_BOOL) { BooleanTensor(it) }
      case(DT_QUINT8, DT_UINT8, DT_QINT8, DT_INT8) { ByteTensor(it) }
      case(DT_BFLOAT16, DT_INT16) { ShortTensor(it) }
      case(DT_INT64) { LongTensor(it) }
      case(DT_STRING) { StringTensor(it) }
    }
    
    operator fun <T : Any> invoke(c_tensor: TF_Tensor) =
        create_switch(TF_TensorType(c_tensor), c_tensor) as Tensor<T>
    
    operator fun invoke(value: Float) = invoke(scalarDimension, f(value))
    operator fun invoke(value: Double) = invoke(scalarDimension, d(value))
    operator fun invoke(value: Boolean) = invoke(scalarDimension, B(value))
    operator fun invoke(value: Byte) = invoke(scalarDimension, b(value))
    operator fun invoke(value: Short) = invoke(scalarDimension, s(value))
    operator fun invoke(value: Int) = invoke(scalarDimension, i(value))
    operator fun invoke(value: Long) = invoke(scalarDimension, l(value))
    operator fun invoke(value: String) = invoke(scalarDimension, a(value))
    
    operator fun invoke(value: FloatArray) = invoke(Shape(value.size), value)
    operator fun invoke(value: DoubleArray) = invoke(Shape(value.size), value)
    operator fun invoke(value: BooleanArray) = invoke(Shape(value.size), value)
    operator fun invoke(value: ByteArray) = invoke(Shape(value.size), value)
    operator fun invoke(value: ShortArray) = invoke(Shape(value.size), value)
    operator fun invoke(value: IntArray) = invoke(Shape(value.size), value)
    operator fun invoke(value: LongArray) = invoke(Shape(value.size), value)
    operator fun invoke(value: Array<String>) = invoke(Shape(value.size), value)
    
    operator fun invoke(shape: Shape, value: FloatArray) = FloatTensor(create(shape, FloatPointer(*value), DT_FLOAT))
    operator fun invoke(shape: Shape, value: DoubleArray) = DoubleTensor(create(shape, DoublePointer(*value), DT_DOUBLE))
    operator fun invoke(shape: Shape, value: BooleanArray) = BooleanTensor(create(shape, BytePointer(*ByteArray(value.size) { if (value[it]) 1 else 0 }), DT_BOOL))
    operator fun invoke(shape: Shape, value: ByteArray) = ByteTensor(create(shape, BytePointer(*value), DT_INT8))
    operator fun invoke(shape: Shape, value: ShortArray) = ShortTensor(create(shape, ShortPointer(*value), DT_INT16))
    operator fun invoke(shape: Shape, value: IntArray) = IntTensor(create(shape, IntPointer(*value), DT_INT32))
    operator fun invoke(shape: Shape, value: LongArray) = LongTensor(create(shape, LongPointer(*value), DT_INT64))
    operator fun invoke(shape: Shape, array: Array<String>): StringTensor {
      val data = TFStringArray.encode(array)
      val t = newTensor(DT_STRING, shape.asLongArray(), data)
      return StringTensor(t, array)
    }
    
    internal fun create(shape: Shape, array: Pointer, dtype: Int) =
        create(shape.asLongArray(), array, dtype)
    
    internal fun create(dims: LongArray, array: Pointer, dtype: Int): TF_Tensor {
      val total = array.sizeof() * array.limit()
      val byteSize = sizeof(dtype) * array.limit()
      return newTensor(dtype, dims, BytePointer(array).capacity(byteSize))
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
  val dtype: DataType<*> = DataType.fromCValue(TF_TensorType(c_tensor))
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
      COMPLEX64, FLOAT -> FloatPointer(ptr).position(0L).capacity((size / 4)).asBuffer()
      DOUBLE -> DoublePointer(ptr).position(0L).capacity((size / 8)).asBuffer()
      QINT32, INT32 -> IntPointer(ptr).position(0L).capacity((size / 4)).asBuffer()
      BOOL, QUINT8, UINT8, QINT8, INT8 -> ptr.position(0L).capacity(size).asBuffer()
      BFLOAT16, INT16 -> ShortPointer(ptr).position(0L).capacity((size / 2)).asBuffer()
      INT64 -> LongPointer(ptr).position(0L).capacity((size / 8)).asBuffer()
      else -> throw IllegalStateException("invalid DataType($dtype)")
    } as B
  }
  
  protected fun copy_tensor(): TF_Tensor {
    val src = TF_TensorData(c_tensor)
    val size = TF_TensorByteSize(c_tensor)
    val t = allocateTensor(dtype.cValue, dims, size)
    val data = TF_TensorData(t)
    memcpy(data, src, size)
    return t
  }
  
  override val size: Int
    get() = numElements.toInt()
}

class FloatTensor(c_tensor: TF_Tensor) : Tensor<Float>(c_tensor) {
  private val buf = createBuffer<FloatBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Float) {
    buf.put(offset, data)
  }
  
  override fun copy() = FloatTensor(copy_tensor())
}

class DoubleTensor(c_tensor: TF_Tensor) : Tensor<Double>(c_tensor) {
  private val buf = createBuffer<DoubleBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Double) {
    buf.put(offset, data)
  }
  
  override fun copy() = DoubleTensor(copy_tensor())
}

class ByteTensor(c_tensor: TF_Tensor) : Tensor<Byte>(c_tensor) {
  private val buf = createBuffer<ByteBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Byte) {
    buf.put(offset, data)
  }
  
  override fun copy() = ByteTensor(copy_tensor())
}

class ShortTensor(c_tensor: TF_Tensor) : Tensor<Short>(c_tensor) {
  private val buf = createBuffer<ShortBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Short) {
    buf.put(offset, data)
  }
  
  override fun copy() = ShortTensor(copy_tensor())
}

class IntTensor(c_tensor: TF_Tensor) : Tensor<Int>(c_tensor) {
  private val buf = createBuffer<IntBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Int) {
    buf.put(offset, data)
  }
  
  override fun copy() = IntTensor(copy_tensor())
}

class LongTensor(c_tensor: TF_Tensor) : Tensor<Long>(c_tensor) {
  private val buf = createBuffer<LongBuffer>()
  override fun get(offset: Int) = buf[offset]
  
  override fun set(offset: Int, data: Long) {
    buf.put(offset, data)
  }
  
  override fun copy() = LongTensor(copy_tensor())
}

class BooleanTensor(c_tensor: TF_Tensor) : Tensor<Boolean>(c_tensor) {
  private val buf = createBuffer<ByteBuffer>()
  override fun get(offset: Int) = buf[offset].toInt() != 0
  
  override fun set(offset: Int, data: Boolean) {
    buf.put(offset, if (data) 1 else 0)
  }
  
  override fun copy() = BooleanTensor(copy_tensor())
}

class StringTensor(private var _c_tensor: TF_Tensor, val array: Array<String>? = null) : Tensor<String>(_c_tensor) {
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
  
  override fun copy() = StringTensor(copy_tensor())
}

object TFStringArray {
  val sizeofUInt64 = Tensor.sizeof(DT_UINT64).toLong()
  
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
      status.check()
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
      status.check()
      p.limit(len.get()).string
    }
  }
}