@file:Suppress("UNCHECKED_CAST")

package wumo.sim.tensorflow.tensor

import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.Pointer.memcpy
import org.bytedeco.javacpp.ShortPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.allocateTensor
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.core.check
import wumo.sim.tensorflow.types.*
import wumo.sim.util.*
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.Buf
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.implementation.ArrayBuf
import wumo.sim.util.ndarray.types.NDType

abstract class Tensor<T : Any>
protected constructor(_c_tensor: TF_Tensor) : Buf<T>() {
  
  companion object {
    fun newTensor(dtype: Int, dims: LongArray, data: Pointer): TF_Tensor {
      return TF_NewTensor(dtype, dims, dims.size, data, data.limit(),
                          object : Deallocator_Pointer_long_Pointer() {
                            override fun call(p0: Pointer?, p1: Long, p2: Pointer?) {
                            }
                          }, null)
    }
    
    fun <T : Any> toNDArray(c_tensor: TF_Tensor): NDArray<T> =
        invoke<T>(c_tensor).toNDArray()
    
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
    
    operator fun invoke(shape: Shape, value: FloatArray): Tensor<Float> =
        FixedSizeTensor(create(shape, FloatPointer(*value), FLOAT))
    
    operator fun invoke(shape: Shape, value: DoubleArray): Tensor<Double> =
        FixedSizeTensor(create(shape, DoublePointer(*value), DOUBLE))
    
    operator fun invoke(shape: Shape, value: BooleanArray): Tensor<Boolean> =
        FixedSizeTensor(create(shape, BytePointer(*ByteArray(value.size) {
          if (value[it]) 1 else 0
        }), BOOL))
    
    operator fun invoke(shape: Shape, value: ByteArray): Tensor<Byte> =
        FixedSizeTensor(create(shape, BytePointer(*value), INT8))
    
    operator fun invoke(shape: Shape, value: ShortArray): Tensor<Short> =
        FixedSizeTensor(create(shape, ShortPointer(*value), INT16))
    
    operator fun invoke(shape: Shape, value: IntArray): Tensor<Int> =
        FixedSizeTensor(create(shape, IntPointer(*value), INT32))
    
    operator fun invoke(shape: Shape, value: LongArray): Tensor<Long> =
        FixedSizeTensor(create(shape, LongPointer(*value), INT64))
    
    operator fun invoke(shape: Shape, array: Array<String>): StringTensor {
      val data = TFStringArray.encode(array)
      val t = newTensor(STRING.cValue, shape.asLongArray()!!, data)
      return StringTensor(t, array)
    }
    
    fun <T : Any> fromNDArray(ndarray: NDArray<T>,
                              dtype: DataType<*> = ndarray.dtype.toDataType()): Tensor<T> {
      val data = when (dtype) {
        is types.STRING -> TFStringArray.encode((dtype.castBuf(ndarray.raw) as ArrayBuf<String>).raw)
        else -> {
          dtype as DataType<T>
          val byteSize = dtype.byteSize
          val ptr = BytePointer((ndarray.numElements * byteSize).toLong())
          for ((i, v) in ndarray.flatten()) {
            dtype.put(ptr, i * byteSize, v)
          }
          ptr
        }
      }
      
      val c_tensor = create(ndarray.shape, data, dtype)
      return createTensor(dtype, c_tensor)
    }
    
    operator fun <T : Any> invoke(c_tensor: TF_Tensor): Tensor<T> {
      val dtype = TF_TensorType(c_tensor).toDataType()
      return createTensor(dtype, c_tensor)
    }
    
    private fun <T : Any> createTensor(dtype: DataType<*>,
                                       c_tensor: TF_Tensor): Tensor<T> {
      val result = when (dtype.baseDataType) {
        is types.STRING -> StringTensor(c_tensor)
        else -> FixedSizeTensor<T>(c_tensor)
      }
      return result as Tensor<T>
    }
    
    internal fun create(shape: Shape, data: Pointer, dtype: DataType<*>): TF_Tensor {
      val bytePointer = data as? BytePointer
          ?: BytePointer(data).capacity(data.sizeof() * data.limit())
      val dims = shape.asLongArray()!!
      return newTensor(dtype.cValue, dims, bytePointer)
    }
    
  }
  
  open val c_tensor: TF_Tensor = _c_tensor
  protected val stride: LongArray
  protected val dims: LongArray
  protected val numElements: Long
  val dtype: DataType<T> = DataType.fromCValue(TF_TensorType(_c_tensor))
  val numDims = TF_NumDims(_c_tensor)
  
  init {
    stride = LongArray(numDims)
    dims = LongArray(numDims) { TF_Dim(_c_tensor, it) }
    var n = 1L
    if (dims.isNotEmpty()) {
      stride[stride.lastIndex] = 1
      for (a in stride.lastIndex - 1 downTo 0)
        stride[a] = dims[a + 1] * stride[a + 1]
      n = dims[0] * stride[0]
    }
    numElements = n
  }
  
  @Suppress("IMPLICIT_CAST_TO_ANY")
  protected fun createBuffer(): BytePointer {
    val data = TF_TensorData(c_tensor)
    val size = TF_TensorByteSize(c_tensor)
    val ptr = BytePointer(data)
    ptr.capacity(size)
    return ptr
  }
  
  protected fun idx(idx: Int): Long = (dtype.byteSize * idx).toLong()
  
  protected fun copy_tensor(): TF_Tensor {
    val src = TF_TensorData(c_tensor)
    val size = TF_TensorByteSize(c_tensor)
    val t = allocateTensor(dtype.cValue, dims, size)
    val data = TF_TensorData(t)
    memcpy(data, src, size)
    return t
  }
  
  fun toNDArray(): NDArray<T> = NDArray(Shape(dims),
                                        dtype.castBuf(this),
                                        dtype.kotlinType.NDType())
  
  override val size: Int
    get() = numElements.toInt()
}

class FixedSizeTensor<T : Any>(c_tensor: TF_Tensor) : Tensor<T>(c_tensor) {
  val byteBuffer = createBuffer()
  
  override fun get(offset: Int): T =
      dtype.get(byteBuffer, offset * dtype.byteSize) as T
  
  override fun set(offset: Int, data: T) {
    dtype.put(byteBuffer, offset * dtype.byteSize, data)
  }
  
  override fun copy(): Buf<T> = FixedSizeTensor(copy_tensor())
  
  override fun slice(start: Int, end: Int): Buf<T> {
    TODO()
  }
}

class StringTensor(private var _c_tensor: TF_Tensor, val array: Array<String>? = null) : Tensor<String>(_c_tensor) {
  override fun slice(start: Int, end: Int): Buf<String> {
    TODO("not implemented")
  }
  
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
  val sizeofUInt64 = UINT64.byteSize.toLong()
  
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