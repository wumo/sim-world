package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.Pointer.memcpy
import org.bytedeco.javacpp.tensorflow.*
import java.lang.reflect.Array
import java.nio.*
import java.util.*

class Tensor(val dtype: DataType, val shape: LongArray) : AutoCloseable {
  override fun close() {
    TF_DeleteTensor(nativeTensor)
  }
  
  lateinit var nativeTensor: TF_Tensor
  
  fun <T> scalarValue(get: (Pointer) -> T): T {
    when {
      TF_NumDims(nativeTensor) != 0 -> throw IllegalStateException("Tensor is not a scalar")
      TF_TensorType(nativeTensor) != dtype.c() -> throw IllegalStateException("Tensor is not a $dtype scalar")
      else -> {
        val data = TF_TensorData(nativeTensor)
        return get(data)
      }
    }
  }
  
  fun numDimensions(): Int {
    return shape.size
  }
  
  fun shape(): LongArray {
    return shape
  }
  
  fun floatValue() = scalarValue { FloatPointer(it).get() }
  fun doubleValue() = scalarValue { DoublePointer(it).get() }
  fun intValue() = scalarValue { IntPointer(it).get() }
  fun LongValue() = scalarValue { LongPointer(it).get() }
  
  fun writeTo(dst: FloatBuffer) {
    if (dtype != DataType.FLOAT)
      throw incompatibleBuffer(dst, dtype)
    val byteBuf = buffer()
    dst.put(byteBuf.asFloatBuffer())
  }
  
  fun writeTo(dst: DoubleBuffer) {
    if (dtype != DataType.DOUBLE)
      throw incompatibleBuffer(dst, dtype)
    val byteBuf = buffer()
    dst.put(byteBuf.asDoubleBuffer())
  }
  
  fun writeTo(dst: IntBuffer) {
    if (dtype != DataType.INT32)
      throw incompatibleBuffer(dst, dtype)
    val byteBuf = buffer()
    dst.put(byteBuf.asIntBuffer())
  }
  
  fun writeTo(dst: LongBuffer) {
    if (dtype != DataType.INT64)
      throw incompatibleBuffer(dst, dtype)
    val byteBuf = buffer()
    dst.put(byteBuf.asLongBuffer())
  }
  
  private fun buffer(): ByteBuffer {
    val data = TF_TensorData(nativeTensor)
    val sz = TF_TensorByteSize(nativeTensor)
    val byteBuf = BytePointer(data)
    byteBuf.capacity(sz)
    return byteBuf.asByteBuffer().order(ByteOrder.nativeOrder())
  }
  
  fun <U : Any> copyTo(dst: U) {
    throwExceptionIfTypeIsIncompatible(dst)
    val num_dims = TF_NumDims(nativeTensor)
    val dtype = DataType.fromC(TF_TensorType(nativeTensor))
    val data = TF_TensorData(nativeTensor)
    val sz = TF_TensorByteSize(nativeTensor)
    if (num_dims == 0) {
      throw IllegalArgumentException("copyTo() is not meant for scalar Tensors, use the scalar " +
                                     "accessor (floatValue(), intValue() etc.) instead")
    }
    if (dtype == DataType.STRING) {
      TODO()
    }
    readNDArray(data, 0, dst, sz, num_dims, dtype)
  }
  
  private fun throwExceptionIfTypeIsIncompatible(o: Any) {
    val rank = numDimensions()
    val oRank = numDimensions(o, dtype)
    if (oRank != rank)
      throw IllegalArgumentException("cannot copy Tensor with $rank dimensions into an object with $oRank")
    if (!objectCompatWithType(o, dtype))
      throw IllegalArgumentException(
          "cannot copy Tensor with DataType $dtype into an object of type ${o.javaClass.name}")
    
    val oShape = LongArray(rank)
    fillShape(o, 0, oShape)
    for (i in oShape.indices) {
      if (oShape[i] != shape()[i]) {
        throw IllegalArgumentException(
            String.format(
                "cannot copy Tensor with shape %s into object with shape %s",
                Arrays.toString(shape()), Arrays.toString(oShape)))
      }
    }
  }
  
  companion object {
    fun fromHandle(nativeTensor: TF_Tensor): Tensor {
      val dtype = TF_TensorType(nativeTensor)
      val shape = LongArray(TF_NumDims(nativeTensor)) {
        TF_Dim(nativeTensor, it)
      }
      return Tensor(DataType.fromC(dtype), shape).apply { this.nativeTensor = nativeTensor }
    }
    
    fun <T> create(obj: Any, type: Class<T>): Tensor {
      val dtype = DataType.fromClass(type)
      if (!objectCompatWithType(obj, dtype))
        throw IllegalArgumentException("DataType of object does not match T (expected $dtype, got ${dataTypeOf(obj)})")
      return create(obj, dtype)
    }
    
    fun create(obj: Any): Tensor {
      return create(obj, dataTypeOf(obj))
    }
    
    private fun objectCompatWithType(obj: Any, dtype: DataType): Boolean {
      val c = baseObjType(obj)
      val dto = dataTypeFromClass(c)
      val nd = numDimensions(obj, dto)
      if (!c.isPrimitive && c != String::class.java && nd != 0)
        throw IllegalArgumentException(
            "cannot create non-scalar Tensors from arrays of boxed values")
      if (dto == dtype)
        return true
      return dto == DataType.STRING && dtype == DataType.UINT8
    }
    
    private fun dataTypeOf(o: Any): DataType {
      val c = baseObjType(o)
      return dataTypeFromClass(c)
    }
    
    private fun baseObjType(o: Any): Class<*> {
      var c: Class<*> = o.javaClass
      while (c.isArray) {
        c = c.componentType
      }
      return c
    }
    
    private fun dataTypeFromClass(c: Class<*>): DataType {
      val ret = DataType.fromClass(c)
      if (ret != DataType.STRING)
        return ret
      throw IllegalArgumentException("cannot create Tensors of type " + c.name)
    }
    
    private fun create(obj: Any, dtype: DataType): Tensor {
      val t = Tensor(dtype, LongArray(numDimensions(obj, dtype)))
      fillShape(obj, 0, t.shape)
      when {
        t.dtype != DataType.STRING -> {
          val sz = elemByteSize(t.dtype)
          val byteSize = sz * numElements(t.shape)
          val num_dims = t.shape.size
          val nativeTensor = TF_AllocateTensor(t.dtype.c(), t.shape, num_dims, byteSize)
          t.nativeTensor = nativeTensor
          val data = TF_TensorData(nativeTensor)
          if (num_dims == 0)
            writeScalar(data, obj, dtype, sz)
          else
            writeNDArray(data, 0, obj, byteSize, num_dims, dtype)
        }
      //TODO 支持String
      }
      return t
    }
    
    private fun writeScalar(data: Pointer, obj: Any, dtype: DataType, sz: Long) {
      when (dtype) {
        DataType.FLOAT -> memcpy(data, FloatPointer(obj as Float), sz)
        DataType.DOUBLE -> memcpy(data, DoublePointer(obj as Double), sz)
        DataType.INT32 -> memcpy(data, IntPointer(obj as Int), sz)
        DataType.UINT8 -> memcpy(data, BytePointer(obj as Byte), sz)
        DataType.INT64 -> memcpy(data, LongPointer(obj as Long), sz)
        DataType.BOOL -> memcpy(data, BytePointer((if (obj as Boolean) 1 else 0).toByte()), sz)
        else -> throw IllegalStateException("invalid DataType($dtype)")
      }
    }
    
    private fun readNDArray(src: Pointer, offset: Long, dst: Any, src_size: Long, dims_left: Int, dtype: DataType): Long {
      return if (dims_left == 1)
        read1DArray(src, offset, dst, src_size, dtype)
      else {
        var sz = 0L
        for (i in 0 until Array.getLength(dst)) {
          val row = Array.get(dst, i)
          sz += readNDArray(src, offset + sz, row, src_size - sz, dims_left - 1, dtype)
        }
        sz
      }
    }
    
    private fun read1DArray(src: Pointer, offset: Long, dst: Any, src_size: Long, dtype: DataType): Long {
      val len = Array.getLength(dst).toLong()
      val size = elemByteSize(dtype)
      val sz = len * size
      if (sz > src_size)
        throw IllegalStateException("cannot fill a Java array of $sz bytes with a Tensor of $src_size bytes")
      src.position<Pointer>(offset / size)//因为下面的各类Pointer的position与src的position单位不同
      when (dtype) {
        DataType.FLOAT -> FloatPointer(src).get(dst as FloatArray)
        DataType.DOUBLE -> DoublePointer(src).get(dst as DoubleArray)
        DataType.INT32 -> IntPointer(src).get(dst as IntArray)
        DataType.UINT8 -> BytePointer(src).get(dst as ByteArray)
        DataType.INT64 -> LongPointer(src).get(dst as LongArray)
        else ->
          throw IllegalStateException("invalid DataType($dtype)")
      }
      
      return sz
    }
    
    private fun writeNDArray(dst: Pointer, offset: Long, obj: Any, dst_size: Long, dims_left: Int, dtype: DataType): Long {
      return if (dims_left == 1)
        write1DArray(dst, offset, obj, dtype)
      else {
        var sz = 0L
        for (i in 0 until Array.getLength(obj)) {
          val row = Array.get(obj, i)
          sz += writeNDArray(dst, offset + sz, row, dst_size - sz, dims_left - 1, dtype)
        }
        sz
      }
    }
    
    private fun write1DArray(data: Pointer, offset: Long, array: Any, dtype: DataType): Long {
      val sz = (Array.getLength(array) * elemByteSize(dtype))
      data.position<Pointer>(offset)
      when (dtype) {
        DataType.FLOAT -> memcpy(data, FloatPointer(*array as FloatArray), sz)
        DataType.DOUBLE -> memcpy(data, DoublePointer(*array as DoubleArray), sz)
        DataType.INT32 -> memcpy(data, IntPointer(*array as IntArray), sz)
        DataType.UINT8 -> memcpy(data, BytePointer(*array as ByteArray), sz)
        DataType.INT64 -> memcpy(data, LongPointer(*array as LongArray), sz)
        DataType.BOOL -> memcpy(data, BytePointer(*array as ByteArray), sz)
        else -> throw IllegalStateException("invalid DataType($dtype)")
      }
      return sz
    }
    
    private fun fillShape(o: Any, dim: Int, shape: LongArray) {
      if (dim == shape.size) return
      val len = Array.getLength(o)
      when {
        len == 0 -> throw IllegalArgumentException("cannot create Tensors with a 0 dimension")
        shape[dim] == 0L -> shape[dim] = len.toLong()
        shape[dim] != len.toLong() ->
          throw IllegalArgumentException(String.format("mismatched lengths (%d and %d) in dimension %d", shape[dim], len, dim))
      }
      for (i in 0 until len)
        fillShape(Array.get(o, i), dim + 1, shape)
    }
    
    private fun numDimensions(o: Any, dtype: DataType): Int {
      val ret = numArrayDimensions(o)
      return if (dtype == DataType.STRING && ret > 0) {
        ret - 1
      } else ret
    }
    
    private fun numArrayDimensions(o: Any): Int {
      var c: Class<*> = o.javaClass
      var i = 0
      while (c.isArray) {
        c = c.componentType
        i++
      }
      return i
    }
    
    private fun numElements(shape: LongArray): Long {
      // assumes a fully-known shape
      var n = 1L
      for (d in shape)
        n *= d
      return n
    }
    
    private fun elemByteSize(dataType: DataType): Long {
      return when (dataType) {
        DataType.FLOAT, DataType.INT32 -> 4
        DataType.DOUBLE, DataType.INT64 -> 8
        DataType.BOOL, DataType.UINT8 -> 1
        DataType.STRING -> throw IllegalArgumentException("STRING tensors do not have a fixed element size")
      }
    }
    
    private fun incompatibleBuffer(buf: Buffer, dataType: DataType): IllegalArgumentException {
      return IllegalArgumentException("cannot use ${buf.javaClass.name} with Tensor of type $dataType")
    }
  }
  
  
}