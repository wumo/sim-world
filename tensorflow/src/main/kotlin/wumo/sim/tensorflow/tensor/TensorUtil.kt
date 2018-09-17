package wumo.sim.tensorflow.tensor

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.Pointer.memcpy
import org.bytedeco.javacpp.tensorflow
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.Buf
import wumo.sim.util.ndarray.NDArray

class BytePointerBuf<T : Any>(
    val bytePointer: BytePointer,
    val dataType: DataType<T>) : Buf<T> {
  
  override fun get(offset: Int): T =
      dataType.get(bytePointer, offset * dataType.byteSize)
  
  override fun set(offset: Int, data: T) {
    dataType.put(bytePointer, offset * dataType.byteSize, data)
  }
  
  override fun copy(): Buf<T> {
    val dst = BytePointer(bytePointer.limit())
    memcpy(dst, bytePointer, bytePointer.limit())
    return BytePointerBuf(dst, dataType)
  }
  
  override fun slice(start: Int, end: Int): Buf<T> {
    TODO("not implemented")
  }
  
  override val size: Int = bytePointer.limit().toInt() / dataType.byteSize
}

/**
 * Returns the constant value of the given tensor, if efficiently calculable.

This function attempts to partially evaluate the given tensor, and
returns its value as a numpy ndarray if this succeeds.

TODO(mrry): Consider whether this function should use a registration
mechanism like gradients and ShapeFunctions, so that it is easily
extensible.

NOTE: If `constant_value(tensor)` returns a non-`None` result, it will no
longer be possible to feed a different value for `tensor`. This allows the
result of this function to influence the graph that is constructed, and
permits static shape optimizations.

Args:
tensor: The Tensor to be evaluated.
partial: If True, the returned numpy array is allowed to have partially
evaluated values. Values that can't be evaluated will be None.

Returns:
A numpy ndarray containing the constant value of the given `tensor`,
or None if it cannot be calculated.
 
 * @see "tensorflow.python.framework.tensor_util.constant_value"
 */
fun constantValue(tensor: Output): NDArray<*>? {
  val result: NDArray<*>? = when (tensor.op.opType) {
    "Const" -> makeNDArray(tensor.op.attrTensor("value"))
    "Shape" -> {
      val inputShape = tensor.op.inputs[0].shape
      if (inputShape.isFullyDefined)
        NDArray(inputShape.asIntArray()!!)
      else
        null
    }
    "Size" -> {
      TODO()
    }
    "Rank" -> {
      TODO()
    }
    "Range" -> {
      TODO()
    }
    "Cast" -> {
      val pre_cast = constantValue(tensor.op.inputs[0]) ?: return null
      val castDtype = tensor.op.attrDataType("DstT")
      
      TODO()
    }
    "Concat" -> {
      TODO()
    }
    "ConcatV2" -> {
      TODO()
    }
    "Pack" -> {
      TODO()
    }
    "Fill" -> TODO()
    "Equal" -> TODO()
    "NotEqual" -> TODO()
    else -> TODO()
  }
  if (result != null)
    tensor.graph.preventFeeding(tensor)
  return result
}

fun makeNDArray(tensor: tensorflow.TensorProto): NDArray<*> {
  val shape = Shape(tensor.tensor_shape().let { dims ->
    IntArray(dims.dim_size()) {
      dims.dim(it).size().toInt()
    }
  })
  
  val num_elements = shape.numElements()
  val tensor_dtype: DataType<*> = DataType.fromCValue(tensor.dtype())
  val dtype = tensor_dtype
  val tensor_content = tensor.tensor_content()
  if (tensor_content != null)
    return NDArray(shape,
                   dtype.castBuf(BytePointerBuf(tensor_content,
                                                dtype as DataType<Any>)),
                   dtype.kotlinType)
  return when (tensor_dtype) {
    FLOAT16, BFLOAT16 ->
      TODO()
    FLOAT -> {
      TODO()
    }
    DOUBLE -> {
      TODO()
    }
    INT32, UINT8, UINT16, INT16, INT8,
    QINT32, QUINT8, QINT8, QINT16, QUINT16 -> {
      val int_val_size = tensor.int_val_size()
      if (int_val_size == 1)
        NDArray(shape, tensor.int_val(0))
      else
        NDArray(shape, IntArray(int_val_size) { tensor.int_val(it) })
    }
    INT64 -> {
      val int64_val_size = tensor.int64_val_size()
      if (int64_val_size == 1)
        NDArray(shape, tensor.int64_val(0))
      else
        NDArray(shape, LongArray(int64_val_size) { tensor.int64_val(it) })
    }
    STRING -> {
      TODO()
    }
    COMPLEX64 -> {
      TODO()
    }
    COMPLEX128 -> {
      TODO()
    }
    BOOL -> {
      TODO()
    }
    else -> error("Unsupported tensor type: $tensor_dtype")
  }
}