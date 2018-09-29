package wumo.sim.tensorflow.types

import org.bytedeco.javacpp.BytePointer
import org.tensorflow.framework.DataType.*
import wumo.sim.util.NONE
import wumo.sim.util.ndarray.types.*

object STRING : types.STRING()
object BOOL : types.BOOL()

object FLOAT16 : types.FLOAT16()
typealias HALF = FLOAT16

object FLOAT : types.FLOAT32()
object DOUBLE : types.FLOAT64()
object BFLOAT16 : types.BFLOAT16()
object COMPLEX64 : types.COMPLEX64()
object COMPLEX128 : types.COMPLEX128()
object INT8 : types.INT8()
object INT16 : types.INT16()
object INT32 : types.INT32()
object INT64 : types.INT64()
object UINT8 : types.UINT8()
object UINT16 : types.UINT16()
object UINT32 : types.UINT32()
object UINT64 : types.UINT64()
object QINT8 : types.QINT8()
object QINT16 : types.QINT16()
object QINT32 : types.QINT32()
object QUINT8 : types.QUINT8()
object QUINT16 : types.QUINT16()
object RESOURCE : types.RESOURCE()
object VARIANT : types.VARIANT()

object STRING_REF : types.STRING_REF()
object BOOL_REF : types.BOOL_REF()
object FLOAT16_REF : types.FLOAT16_REF()
object FLOAT_REF : types.FLOAT_REF()
object DOUBLE_REF : types.DOUBLE_REF()
object BFLOAT16_REF : types.BFLOAT16_REF()
object COMPLEX64_REF : types.COMPLEX64_REF()
object COMPLEX128_REF : types.COMPLEX128_REF()
object INT8_REF : types.INT8_REF()
object INT16_REF : types.INT16_REF()
object INT32_REF : types.INT32_REF()
object INT64_REF : types.INT64_REF()
object UINT8_REF : types.UINT8_REF()
object UINT16_REF : types.UINT16_REF()
object UINT32_REF : types.UINT32_REF()
object UINT64_REF : types.UINT64_REF()
object QINT8_REF : types.QINT8_REF()
object QINT16_REF : types.QINT16_REF()
object QINT32_REF : types.QINT32_REF()
object QUINT8_REF : types.QUINT8_REF()
object QUINT16_REF : types.QUINT16_REF()
object RESOURCE_REF : types.RESOURCE_REF()
object VARIANT_REF : types.VARIANT_REF()

interface ReducibleDataType<KotlinType : Any> : DataType<KotlinType>
interface NumericDataType<KotlinType : Any> : ReducibleDataType<KotlinType>
interface NonQuantizedDataType<KotlinType : Any> : NumericDataType<KotlinType>
interface MathDataType<KotlinType : Any> : NonQuantizedDataType<KotlinType>
interface RealDataType<KotlinType : Any> : MathDataType<KotlinType>
interface ComplexDataType<KotlinType : Any> : MathDataType<KotlinType>
interface Int32OrInt64OrFloat16OrFloat32OrFloat64<KotlinType : Any> : RealDataType<KotlinType>
interface IntOrUInt<KotlinType : Any> : RealDataType<KotlinType>
interface UInt8OrInt32OrInt64<KotlinType : Any> : IntOrUInt<KotlinType>, Int32OrInt64OrFloat16OrFloat32OrFloat64<KotlinType>
interface Int32OrInt64<KotlinType : Any> : UInt8OrInt32OrInt64<KotlinType>
interface DecimalDataType<KotlinType : Any> : RealDataType<KotlinType>
interface BFloat16OrFloat32OrFloat64<KotlinType : Any> : DecimalDataType<KotlinType>
interface BFloat16OrFloat16OrFloat32<KotlinType : Any> : DecimalDataType<KotlinType>
interface Float16OrFloat32OrFloat64<KotlinType : Any> : DecimalDataType<KotlinType>, Int32OrInt64OrFloat16OrFloat32OrFloat64<KotlinType>, BFloat16OrFloat32OrFloat64<KotlinType>
interface Float32OrFloat64<KotlinType : Any> : Float16OrFloat32OrFloat64<KotlinType>
interface Int32OrInt64OrFloat32OrFloat64<KotlinType : Any> : Float32OrFloat64<KotlinType>, Int32OrInt64<KotlinType>
interface QuantizedDataType<KotlinType : Any> : NumericDataType<KotlinType>

abstract class DataTypeComparator<T : Any> : DataType<T> {
  override fun toString() = name
  override fun equals(that: Any?) =
      if (that is DataType<*>) cValue == that.cValue
      else false
  
  override fun hashCode() = cValue
}

object types {
  open class STRING : DataTypeComparator<String>(), ReducibleDataType<String> {
    override val byteSize = -1
    override val priority = 1000
    
    override fun zero() = ""
    override fun one() = NONE()
    
    override val protoType = DT_STRING
    override val ndtype: NDType<String> = NDString
    
  }
  
  open class BOOL : DataTypeComparator<Boolean>(), ReducibleDataType<Boolean> {
    override val byteSize = 1
    override val priority = 0
    
    override fun zero() = false
    override fun one() = true
    
    override val protoType = DT_BOOL
    override val ndtype: NDType<Boolean> = NDBool
    
    override fun put(buffer: BytePointer, idx: Int, element: Boolean) {
      buffer.putBool(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Boolean {
      return buffer.getBool(idx.toLong())
    }
  }
  
  open class FLOAT16 : DataTypeComparator<Float>(), Float16OrFloat32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    override val byteSize = 2
    override val priority = -1
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_HALF
    override val ndtype: NDType<Float> = NDFloat
  }
  
  open class FLOAT32 : DataTypeComparator<Float>(), Float32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    override val byteSize = 4
    override val priority = 220
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_FLOAT
    override val ndtype: NDType<Float> = NDFloat
    
    override fun put(buffer: BytePointer, idx: Int, element: Float) {
      buffer.putFloat(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Float {
      return buffer.getFloat(idx.toLong())
    }
  }
  
  open class FLOAT64 : DataTypeComparator<Double>(), Float32OrFloat64<Double> {
    
    override val byteSize = 8
    override val priority = 230
    
    override fun zero() = 0.0
    override fun one() = 1.0
    
    override val protoType = DT_DOUBLE
    override val ndtype: NDType<Double> = NDDouble
    
    override fun put(buffer: BytePointer, idx: Int, element: Double) {
      buffer.putDouble(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Double {
      return buffer.getDouble(idx.toLong())
    }
  }
  
  open class BFLOAT16 : DataTypeComparator<Float>(), BFloat16OrFloat32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    
    override val byteSize = 2
    override val priority = -1
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_BFLOAT16
    override val ndtype: NDType<Float> = NDFloat
  }
  
  open class COMPLEX64 : DataTypeComparator<Double>(), ComplexDataType<Double> {
    
    override val byteSize = 8
    override val priority = -1
    
    override fun zero() = NONE()
    override fun one() = NONE()
    
    override val protoType = DT_COMPLEX64
    override val ndtype: NDType<Double> = NDDouble
  }
  
  open class COMPLEX128 : DataTypeComparator<Double>(), ComplexDataType<Double> {
    
    override val byteSize = 16
    override val priority = -1
    
    override fun zero() = NONE()
    override fun one() = NONE()
    
    override val protoType = DT_COMPLEX128
    override val ndtype: NDType<Double> = NDDouble
  }
  
  open class INT8 : DataTypeComparator<Byte>(), IntOrUInt<Byte> {
    
    override val byteSize = 1
    override val priority = 40
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_INT8
    override val ndtype: NDType<Byte> = NDByte
    
    override fun put(buffer: BytePointer, idx: Int, element: Byte) {
      buffer.put(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Byte {
      return buffer.get(idx.toLong())
    }
  }
  
  open class INT16 : DataTypeComparator<Short>(), IntOrUInt<Short> {
    
    override val byteSize = 2
    override val priority = 80
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_INT16
    override val ndtype: NDType<Short> = NDShort
    
    override fun put(buffer: BytePointer, idx: Int, element: Short) {
      buffer.putShort(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Short {
      return buffer.getShort(idx.toLong())
    }
  }
  
  open class INT32 : DataTypeComparator<Int>(), Int32OrInt64<Int> {
    
    override val byteSize = 4
    override val priority = 100
    
    override fun zero() = 0
    override fun one() = 1
    
    override val protoType = DT_INT32
    override val ndtype: NDType<Int> = NDInt
    
    override fun put(buffer: BytePointer, idx: Int, element: Int) {
      buffer.putInt(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Int {
      return buffer.getInt(idx.toLong())
    }
  }
  
  open class INT64 : DataTypeComparator<Long>(), Int32OrInt64<Long> {
    
    override val byteSize = 8
    override val priority = 110
    
    override fun zero() = 0L
    override fun one() = 1L
    
    override val protoType = DT_INT64
    override val ndtype: NDType<Long> = NDLong
    
    override fun put(buffer: BytePointer, idx: Int, element: Long) {
      buffer.putLong(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Long {
      return buffer.getLong(idx.toLong())
    }
  }
  
  open class UINT8 : DataTypeComparator<Byte>(), UInt8OrInt32OrInt64<Byte> {
    
    override val byteSize = 1
    override val priority = 20
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_UINT8
    override val ndtype: NDType<Byte> = NDByte
    
    override fun put(buffer: BytePointer, idx: Int, element: Byte) {
      buffer.put(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Byte {
      return buffer.get(idx.toLong())
    }
  }
  
  open class UINT16 : DataTypeComparator<Short>(), IntOrUInt<Short> {
    
    override val byteSize = 2
    override val priority = 60
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_UINT16
    override val ndtype: NDType<Short> = NDShort
    
    override fun put(buffer: BytePointer, idx: Int, element: Short) {
      buffer.putShort(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Short {
      return buffer.getShort(idx.toLong())
    }
  }
  
  open class UINT32 : DataTypeComparator<Long>(), IntOrUInt<Long> {
    
    override val byteSize = 4
    override val priority = 85
    
    override fun zero(): Long = 0
    override fun one(): Long = 1
    
    override val protoType = DT_UINT32
    override val ndtype: NDType<Long> = NDLong
    override fun toString() = name
    
    override fun put(buffer: BytePointer, idx: Int, element: Long) {
      buffer.putLong(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Long {
      return buffer.getLong(idx.toLong())
    }
  }
  
  open class UINT64 : DataTypeComparator<Long>(), IntOrUInt<Long> {
    
    override val byteSize = 8
    override val priority = 105
    
    override fun zero() = NONE()
    override fun one() = NONE()
    
    override val protoType = DT_UINT64
    override val ndtype: NDType<Long> = NDLong
    
  }
  
  open class QINT8 : DataTypeComparator<Byte>(), QuantizedDataType<Byte> {
    
    override val byteSize = 1
    override val priority = 30
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_QINT8
    override val ndtype: NDType<Byte> = NDByte
    override fun toString() = name
    
    override fun put(buffer: BytePointer, idx: Int, element: Byte) {
      buffer.put(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Byte {
      return buffer.get(idx.toLong())
    }
  }
  
  open class QINT16 : DataTypeComparator<Short>(), QuantizedDataType<Short> {
    
    override val byteSize = 2
    override val priority = 70
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_QINT16
    override val ndtype: NDType<Short> = NDShort
    
    override fun put(buffer: BytePointer, idx: Int, element: Short) {
      buffer.putShort(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Short {
      return buffer.getShort(idx.toLong())
    }
  }
  
  open class QINT32 : DataTypeComparator<Int>(), QuantizedDataType<Int> {
    
    override val byteSize = 4
    override val priority = 90
    
    override fun zero(): Int = 0
    override fun one(): Int = 1
    
    override val protoType = DT_QINT32
    override val ndtype: NDType<Int> = NDInt
    
    override fun put(buffer: BytePointer, idx: Int, element: Int) {
      buffer.putInt(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Int {
      return buffer.getInt(idx.toLong())
    }
  }
  
  open class QUINT8 : DataTypeComparator<Byte>(), QuantizedDataType<Byte> {
    
    override val byteSize = 1
    override val priority = 10
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_QUINT8
    override val ndtype: NDType<Byte> = NDByte
    
    override fun put(buffer: BytePointer, idx: Int, element: Byte) {
      buffer.put(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Byte {
      return buffer.get(idx.toLong())
    }
  }
  
  open class QUINT16 : DataTypeComparator<Short>(), QuantizedDataType<Short> {
    
    override val byteSize = 2
    override val priority = 50
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_QUINT16
    override val ndtype: NDType<Short> = NDShort
    
    override fun put(buffer: BytePointer, idx: Int, element: Short) {
      buffer.putShort(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Short {
      return buffer.getShort(idx.toLong())
    }
  }
  
  open class RESOURCE : DataTypeComparator<Long>(), DataType<Long> {
    
    override val byteSize = -1
    override val priority = -1
    
    override fun zero() = NONE()
    override fun one() = NONE()
    
    override val protoType = DT_RESOURCE
    override val ndtype: NDType<Long> = NDLong
    
    override fun put(buffer: BytePointer, idx: Int, element: Long) {
      buffer.putLong(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Long {
      return buffer.getLong(idx.toLong())
    }
  }
  
  open class VARIANT : DataTypeComparator<Long>(), DataType<Long> {
    
    override val byteSize = -1
    override val priority = -1
    
    override fun zero() = NONE()
    override fun one() = NONE()
    
    override val protoType = DT_VARIANT
    override val ndtype: NDType<Long> = NDLong
    override fun toString() = name
    
    override fun put(buffer: BytePointer, idx: Int, element: Long) {
      buffer.putLong(idx.toLong(), element)
    }
    
    override fun get(buffer: BytePointer, idx: Int): Long {
      return buffer.getLong(idx.toLong())
    }
  }
  
  open class STRING_REF : STRING() {
    override val protoType = DT_STRING_REF
  }
  
  open class BOOL_REF : BOOL() {
    override val protoType = DT_BOOL_REF
  }
  
  open class FLOAT16_REF : FLOAT16() {
    override val protoType = DT_HALF_REF
  }
  
  open class FLOAT_REF : FLOAT32() {
    
    override val protoType = DT_FLOAT_REF
  }
  
  open class DOUBLE_REF : FLOAT64() {
    
    override val protoType = DT_DOUBLE_REF
  }
  
  open class BFLOAT16_REF : BFLOAT16() {
    
    override val protoType = DT_BFLOAT16_REF
  }
  
  open class COMPLEX64_REF : COMPLEX64() {
    
    override val protoType = DT_COMPLEX64_REF
  }
  
  open class COMPLEX128_REF : COMPLEX128() {
    
    override val protoType = DT_COMPLEX128_REF
  }
  
  open class INT8_REF : INT8() {
    override val protoType = DT_INT8_REF
  }
  
  open class INT16_REF : INT16() {
    override val protoType = DT_INT16_REF
  }
  
  open class INT32_REF : INT32() {
    override val protoType = DT_INT32_REF
  }
  
  open class INT64_REF : INT64() {
    override val protoType = DT_INT64_REF
  }
  
  open class UINT8_REF : UINT8() {
    
    override val protoType = DT_UINT8_REF
  }
  
  open class UINT16_REF : UINT16() {
    override val protoType = DT_UINT16_REF
  }
  
  open class UINT32_REF : UINT32() {
    override val protoType = DT_UINT32_REF
  }
  
  open class UINT64_REF : UINT64() {
    override val protoType = DT_UINT64_REF
  }
  
  open class QINT8_REF : QINT8() {
    override val protoType = DT_QINT8_REF
  }
  
  open class QINT16_REF : QINT16() {
    
    override val protoType = DT_QINT16_REF
  }
  
  open class QINT32_REF : QINT32() {
    override val protoType = DT_QINT32_REF
  }
  
  open class QUINT8_REF : QUINT8() {
    override val protoType = DT_QUINT8_REF
  }
  
  open class QUINT16_REF : QUINT16() {
    
    override val protoType = DT_QUINT16_REF
  }
  
  open class RESOURCE_REF : RESOURCE() {
    override val protoType = DT_RESOURCE_REF
  }
  
  open class VARIANT_REF : VARIANT() {
    override val protoType = DT_VARIANT_REF
  }
}