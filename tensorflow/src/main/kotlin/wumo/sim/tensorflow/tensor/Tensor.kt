@file:Suppress("UNCHECKED_CAST")

package wumo.sim.tensorflow.tensor

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.LongPointer
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.Pointer.memcpy
import org.bytedeco.javacpp.SizeTPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.allocateTensor
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.core.check
import wumo.sim.tensorflow.types.*
import wumo.sim.util.NONE
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.BytePointerBuf
import wumo.sim.util.scalarDimension

open class Tensor<T : Any>(_c_tensor: TF_Tensor,
                           val dtype: DataType<T>)
  : BytePointerBuf<T>(_c_tensor.dataPtr(), dtype.ndtype) {
  
  companion object {
    fun newTensor(dtype: Int, dims: LongArray, data: Pointer): TF_Tensor {
      return TF_NewTensor(dtype, dims, dims.size, data, data.limit(),
                          object : Deallocator_Pointer_long_Pointer() {
                            override fun call(p0: Pointer?, p1: Long, p2: Pointer?) {
                            }
                          }, null)
    }
    
    fun TF_Tensor.dataPtr(): BytePointer {
      val data = TF_TensorData(this)
      val size = TF_TensorByteSize(this)
      val ptr = BytePointer(data)
      ptr.capacity(size)
      return ptr
    }
    
    fun <T : Any> toNDArray(c_tensor: TF_Tensor): NDArray<T> {
      val dtype = TF_TensorType(c_tensor).toDataType<T>()
      val tensor = Tensor(c_tensor, dtype)
      return tensor.toNDArray()
    }
    
    operator fun <T : Any> invoke(_c_tensor: TF_Tensor, dtype: DataType<T>): Tensor<T> =
        Tensor(_c_tensor, dtype)
    
    operator fun invoke(value: Float) = invoke(scalarDimension, FLOAT) { value }
    operator fun invoke(value: Double) = invoke(scalarDimension, DOUBLE) { value }
    operator fun invoke(value: Boolean) = invoke(scalarDimension, BOOL) { value }
    operator fun invoke(value: Byte) = invoke(scalarDimension, INT8) { value }
    operator fun invoke(value: Short) = invoke(scalarDimension, INT16) { value }
    operator fun invoke(value: Int) = invoke(scalarDimension, INT32) { value }
    operator fun invoke(value: Long) = invoke(scalarDimension, INT64) { value }
    operator fun invoke(value: String) = invoke(scalarDimension, STRING) { value }
    
    operator fun invoke(value: FloatArray) = invoke(Shape(value.size), FLOAT) { value[it] }
    operator fun invoke(value: DoubleArray) = invoke(Shape(value.size), DOUBLE) { value[it] }
    operator fun invoke(value: BooleanArray) = invoke(Shape(value.size), BOOL) { value[it] }
    operator fun invoke(value: ByteArray) = invoke(Shape(value.size), INT8) { value[it] }
    operator fun invoke(value: ShortArray) = invoke(Shape(value.size), INT16) { value[it] }
    operator fun invoke(value: IntArray) = invoke(Shape(value.size), INT32) { value[it] }
    operator fun invoke(value: LongArray) = invoke(Shape(value.size), INT64) { value[it] }
    operator fun invoke(value: Array<String>) = invoke(Shape(value.size), STRING) { value[it] }
    
    operator fun invoke(shape: Shape, value: FloatArray) = invoke(shape, FLOAT) { value[it] }
    operator fun invoke(shape: Shape, value: DoubleArray) = invoke(shape, DOUBLE) { value[it] }
    operator fun invoke(shape: Shape, value: BooleanArray) = invoke(shape, BOOL) { value[it] }
    operator fun invoke(shape: Shape, value: ByteArray) = invoke(shape, INT8) { value[it] }
    operator fun invoke(shape: Shape, value: ShortArray) = invoke(shape, INT16) { value[it] }
    operator fun invoke(shape: Shape, value: IntArray) = invoke(shape, INT32) { value[it] }
    operator fun invoke(shape: Shape, value: LongArray) = invoke(shape, INT64) { value[it] }
    
    operator fun invoke(shape: Shape, array: Array<String>): StringTensor {
      val data = TFStringArray.encode(array)
      val t = newTensor(STRING.cValue, shape.asLongArray()!!, data)
      return StringTensor(t, array)
    }
    
    inline operator fun <T : Any> invoke(shape: Shape, dtype: DataType<T>, init: (Int) -> T): Tensor<T> {
      val size = shape.numElements()
      val byteSize = dtype.byteSize
      val data = BytePointer((size * byteSize).toLong())
      for (i in 0 until size)
        dtype.put(data, i * byteSize, init(i))
      return Tensor(newTensor(dtype.cValue, shape.asLongArray()!!, data), dtype)
    }
    
    fun <T : Any> fromNDArray(ndarray: NDArray<T>): Tensor<T> {
      val src = when (val buf = ndarray.raw) {
        is BytePointerBuf<T> -> BytePointer(buf.ptr)
        else -> NONE()
      }
      val dtype = ndarray.dtype.toDataType()
      return Tensor(newTensor(dtype.cValue, ndarray.shape.asLongArray()!!, src), dtype)
    }
    
    fun <T : Any, R : Any> fromNDArray(ndarray: NDArray<T>, dtype: DataType<R>): Tensor<R> {
      val srcDtype = ndarray.dtype.toDataType()
      val src = when (val buf = ndarray.raw) {
        is BytePointerBuf<T> -> BytePointer(buf.ptr)
        else -> NONE()
      }
      val dst = if (srcDtype == dtype) src
      else {
        val byteSize = dtype.byteSize
        val data = BytePointer((ndarray.size * byteSize).toLong())
        val ndtype = dtype.ndtype
        for (i in 0 until ndarray.size)
          dtype.put(data, i * byteSize, ndtype.cast(srcDtype.get(src, i)))
        data
      }
      return Tensor(newTensor(dtype.cValue, ndarray.shape.asLongArray()!!, dst), dtype)
    }
  }
  
  open val c_tensor: TF_Tensor = _c_tensor
  protected val stride: LongArray
  protected val dims: LongArray
  protected val numElements: Long
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
  
  val byteBuffer = createBuffer()
  
  fun toNDArray(): NDArray<T> = NDArray(Shape(dims), BytePointerBuf(byteBuffer, ndType))
  
  override fun get(idx: Int): T =
      dtype.get(byteBuffer, idx * dtype.byteSize)
  
  override fun set(idx: Int, data: T) {
    dtype.put(byteBuffer, idx * dtype.byteSize, data)
  }
  
  override fun copy(): BytePointerBuf<T> = Tensor(copy_tensor(), dtype)
  
  override fun slice(start: Int, end: Int): BytePointerBuf<T> {
    TODO()
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

class StringTensor(private var _c_tensor: TF_Tensor,
                   val array: Array<String>? = null)
  : Tensor<String>(_c_tensor, STRING) {
  
  override fun slice(start: Int, end: Int): BytePointerBuf<String> {
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
  
  override fun get(idx: Int) = buf[idx]
  
  override fun set(idx: Int, data: String) {
    buf[idx] = data
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