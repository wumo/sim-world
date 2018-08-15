package wumo.sim.tensorflow.types

import org.tensorflow.framework.DataType.*
import wumo.sim.util.NOPE

typealias STRING = types.STRING
typealias BOOL = types.BOOL
typealias HALF = FLOAT16
typealias FLOAT16 = types.FLOAT16
typealias FLOAT = types.FLOAT
typealias DOUBLE = types.FLOAT64
typealias BFLOAT16 = types.BFLOAT16
typealias COMPLEX64 = types.COMPLEX64
typealias COMPLEX128 = types.COMPLEX128
typealias INT8 = types.INT8
typealias INT16 = types.INT16
typealias INT32 = types.INT32
typealias INT64 = types.INT64
typealias UINT8 = types.UINT8
typealias UINT16 = types.UINT16
typealias UINT32 = types.UINT32
typealias UINT64 = types.UINT64
typealias QINT8 = types.QINT8
typealias QINT16 = types.QINT16
typealias QINT32 = types.QINT32
typealias QUINT8 = types.QUINT8
typealias QUINT16 = types.QUINT16
typealias RESOURCE = types.RESOURCE
typealias VARIANT = types.VARIANT

typealias STRING_REF = types.STRING_REF
typealias BOOL_REF = types.BOOL_REF
typealias FLOAT16_REF = types.FLOAT16_REF
typealias FLOAT_REF = types.FLOAT_REF
typealias DOUBLE_REF = types.DOUBLE_REF
typealias BFLOAT16_REF = types.BFLOAT16_REF
typealias COMPLEX64_REF = types.COMPLEX64_REF
typealias COMPLEX128_REF = types.COMPLEX128_REF
typealias INT8_REF = types.INT8_REF
typealias INT16_REF = types.INT16_REF
typealias INT32_REF = types.INT32_REF
typealias INT64_REF = types.INT64_REF
typealias UINT8_REF = types.UINT8_REF
typealias UINT16_REF = types.UINT16_REF
typealias UINT32_REF = types.UINT32_REF
typealias UINT64_REF = types.UINT64_REF
typealias QINT8_REF = types.QINT8_REF
typealias QINT16_REF = types.QINT16_REF
typealias QINT32_REF = types.QINT32_REF
typealias QUINT8_REF = types.QUINT8_REF
typealias QUINT16_REF = types.QUINT16_REF
typealias RESOURCE_REF = types.RESOURCE_REF
typealias VARIANT_REF = types.VARIANT_REF

object types {
  interface ReducibleDataType<KotlinType> : DataType<KotlinType>
  interface NumericDataType<KotlinType> : ReducibleDataType<KotlinType>
  interface NonQuantizedDataType<KotlinType> : NumericDataType<KotlinType>
  interface MathDataType<KotlinType> : NonQuantizedDataType<KotlinType>
  interface RealDataType<KotlinType> : MathDataType<KotlinType>
  interface ComplexDataType<KotlinType> : MathDataType<KotlinType>
  interface Int32OrInt64OrFloat16OrFloat32OrFloat64<KotlinType> : RealDataType<KotlinType>
  interface IntOrUInt<KotlinType> : RealDataType<KotlinType>
  interface UInt8OrInt32OrInt64<KotlinType> : IntOrUInt<KotlinType>, Int32OrInt64OrFloat16OrFloat32OrFloat64<KotlinType>
  interface Int32OrInt64<KotlinType> : UInt8OrInt32OrInt64<KotlinType>
  interface DecimalDataType<KotlinType> : RealDataType<KotlinType>
  interface BFloat16OrFloat32OrFloat64<KotlinType> : DecimalDataType<KotlinType>
  interface BFloat16OrFloat16OrFloat32<KotlinType> : DecimalDataType<KotlinType>
  interface Float16OrFloat32OrFloat64<KotlinType> : DecimalDataType<KotlinType>, Int32OrInt64OrFloat16OrFloat32OrFloat64<KotlinType>, BFloat16OrFloat32OrFloat64<KotlinType>
  interface Float32OrFloat64<KotlinType> : Float16OrFloat32OrFloat64<KotlinType>
  interface Int32OrInt64OrFloat32OrFloat64<KotlinType> : Float32OrFloat64<KotlinType>, Int32OrInt64<KotlinType>
  interface QuantizedDataType<KotlinType> : NumericDataType<KotlinType>
  
  abstract class DataTypeComparator<T> : DataType<T> {
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object STRING : DataTypeComparator<String>(), ReducibleDataType<String> {
    
    override val byteSize = -1
    override val priority = 1000
    
    override fun zero() = ""
    override fun one() = NOPE()
    
    override val protoType = DT_STRING
  }
  
  object BOOL : DataTypeComparator<Boolean>(), ReducibleDataType<Boolean> {
    override val byteSize = 1
    override val priority = 0
    
    override fun zero() = false
    override fun one() = true
    
    override val protoType = DT_BOOL
  }
  
  object FLOAT16 : DataTypeComparator<Float>(), Float16OrFloat32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    override val byteSize = 2
    override val priority = -1
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_HALF
  }
  
  object FLOAT : DataTypeComparator<Float>(), Float32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    override val byteSize = 4
    override val priority = 220
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_FLOAT
  }
  
  object FLOAT64 : DataTypeComparator<Double>(), Float32OrFloat64<Double> {
    
    override val byteSize = 8
    override val priority = 230
    
    override fun zero() = 0.0
    override fun one() = 1.0
    
    override val protoType = DT_DOUBLE
  }
  
  object BFLOAT16 : DataTypeComparator<Float>(), BFloat16OrFloat32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    
    override val byteSize = 2
    override val priority = -1
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_BFLOAT16
  }
  
  object COMPLEX64 : DataTypeComparator<Double>(), ComplexDataType<Double> {
    
    override val byteSize = 8
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_COMPLEX64
  }
  
  object COMPLEX128 : DataTypeComparator<Double>(), ComplexDataType<Double> {
    
    override val byteSize = 16
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_COMPLEX128
  }
  
  object INT8 : DataTypeComparator<Byte>(), IntOrUInt<Byte> {
    
    override val byteSize = 1
    override val priority = 40
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_INT8
  }
  
  object INT16 : DataTypeComparator<Short>(), IntOrUInt<Short> {
    
    override val byteSize = 2
    override val priority = 80
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_INT16
  }
  
  object INT32 : DataTypeComparator<Int>(), Int32OrInt64<Int> {
    
    override val byteSize = 4
    override val priority = 100
    
    override fun zero() = 0
    override fun one() = 1
    
    override val protoType = DT_INT32
  }
  
  object INT64 : DataTypeComparator<Long>(), Int32OrInt64<Long> {
    
    override val byteSize = 8
    override val priority = 110
    
    override fun zero() = 0L
    override fun one() = 1L
    
    override val protoType = DT_INT64
  }
  
  object UINT8 : DataTypeComparator<Byte>(), UInt8OrInt32OrInt64<Byte> {
    
    override val byteSize = 1
    override val priority = 20
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_UINT8
  }
  
  object UINT16 : DataTypeComparator<Short>(), IntOrUInt<Short> {
    
    override val byteSize = 2
    override val priority = 60
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_UINT16
  }
  
  object UINT32 : DataTypeComparator<Long>(), IntOrUInt<Long> {
    
    override val byteSize = 4
    override val priority = 85
    
    override fun zero(): Long = 0
    override fun one(): Long = 1
    
    override val protoType = DT_UINT32
    override fun toString() = name
  }
  
  object UINT64 : DataTypeComparator<Long>(), IntOrUInt<Long> {
    
    override val byteSize = 8
    override val priority = 105
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_UINT64
  }
  
  object QINT8 : DataTypeComparator<Byte>(), QuantizedDataType<Byte> {
    
    override val byteSize = 1
    override val priority = 30
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_QINT8
    override fun toString() = name
  }
  
  object QINT16 : DataTypeComparator<Short>(), QuantizedDataType<Short> {
    
    override val byteSize = 2
    override val priority = 70
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_QINT16
  }
  
  object QINT32 : DataTypeComparator<Int>(), QuantizedDataType<Int> {
    
    override val byteSize = 4
    override val priority = 90
    
    override fun zero(): Int = 0
    override fun one(): Int = 1
    
    override val protoType = DT_QINT32
  }
  
  object QUINT8 : DataTypeComparator<Byte>(), QuantizedDataType<Byte> {
    
    override val byteSize = 1
    override val priority = 10
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_QUINT8
  }
  
  object QUINT16 : DataTypeComparator<Short>(), QuantizedDataType<Short> {
    
    override val byteSize = 2
    override val priority = 50
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_QUINT16
  }
  
  object RESOURCE : DataTypeComparator<Long>(), DataType<Long> {
    
    override val byteSize = -1
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_RESOURCE
  }
  
  object VARIANT : DataTypeComparator<Long>(), DataType<Long> {
    
    override val byteSize = -1
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_VARIANT
    override fun toString() = name
  }
  
  object STRING_REF : DataTypeComparator<String>(), ReducibleDataType<String> {
    
    override val byteSize = -1
    override val priority = 1000
    
    override fun zero() = ""
    override fun one() = NOPE()
    
    override val protoType = DT_STRING_REF
  }
  
  object BOOL_REF : DataTypeComparator<Boolean>(), ReducibleDataType<Boolean> {
    
    override val byteSize = 1
    override val priority = 0
    
    override fun zero() = false
    override fun one() = true
    
    override val protoType = DT_BOOL_REF
  }
  
  object FLOAT16_REF : DataTypeComparator<Float>(), Float16OrFloat32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    
    override val byteSize = 2
    override val priority = -1
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_HALF_REF
  }
  
  object FLOAT_REF : DataTypeComparator<Float>(), Float32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    
    override val byteSize = 4
    override val priority = 220
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_FLOAT_REF
  }
  
  object DOUBLE_REF : DataTypeComparator<Double>(), Float32OrFloat64<Double> {
    
    override val byteSize = 8
    override val priority = 230
    
    override fun zero() = 0.0
    override fun one() = 1.0
    
    override val protoType = DT_DOUBLE_REF
  }
  
  object BFLOAT16_REF : DataTypeComparator<Float>(), BFloat16OrFloat32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    
    override val byteSize = 2
    override val priority = -1
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = DT_BFLOAT16_REF
  }
  
  object COMPLEX64_REF : DataTypeComparator<Double>(), ComplexDataType<Double> {
    
    override val byteSize = 8
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_COMPLEX64_REF
  }
  
  object COMPLEX128_REF : DataTypeComparator<Double>(), ComplexDataType<Double> {
    
    override val byteSize = 16
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_COMPLEX128_REF
  }
  
  object INT8_REF : DataTypeComparator<Byte>(), IntOrUInt<Byte> {
    
    override val byteSize = 1
    override val priority = 40
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_INT8_REF
  }
  
  object INT16_REF : DataTypeComparator<Short>(), IntOrUInt<Short> {
    
    override val byteSize = 2
    override val priority = 80
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_INT16_REF
  }
  
  object INT32_REF : DataTypeComparator<Int>(), Int32OrInt64<Int> {
    
    override val byteSize = 4
    override val priority = 100
    
    override fun zero() = 0
    override fun one() = 1
    
    override val protoType = DT_INT32_REF
  }
  
  object INT64_REF : DataTypeComparator<Long>(), Int32OrInt64<Long> {
    
    override val byteSize = 8
    override val priority = 110
    
    override fun zero() = 0L
    override fun one() = 1L
    
    override val protoType = DT_INT64_REF
  }
  
  object UINT8_REF : DataTypeComparator<Byte>(), UInt8OrInt32OrInt64<Byte> {
    
    override val byteSize = 1
    override val priority = 20
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_UINT8_REF
  }
  
  object UINT16_REF : DataTypeComparator<Short>(), IntOrUInt<Short> {
    
    override val byteSize = 2
    override val priority = 60
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_UINT16_REF
  }
  
  object UINT32_REF : DataTypeComparator<Long>(), IntOrUInt<Long> {
    
    override val byteSize = 4
    override val priority = 85
    
    override fun zero(): Long = 0
    override fun one(): Long = 1
    
    override val protoType = DT_UINT32_REF
  }
  
  object UINT64_REF : DataTypeComparator<Long>(), IntOrUInt<Long> {
    
    override val byteSize = 8
    override val priority = 105
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_UINT64_REF
  }
  
  object QINT8_REF : DataTypeComparator<Byte>(), QuantizedDataType<Byte> {
    
    override val byteSize = 1
    override val priority = 30
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_QINT8_REF
  }
  
  object QINT16_REF : DataTypeComparator<Short>(), QuantizedDataType<Short> {
    
    override val byteSize = 2
    override val priority = 70
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_QINT16_REF
  }
  
  object QINT32_REF : DataTypeComparator<Int>(), QuantizedDataType<Int> {
    
    override val byteSize = 4
    override val priority = 90
    
    override fun zero(): Int = 0
    override fun one(): Int = 1
    
    override val protoType = DT_QINT32_REF
  }
  
  object QUINT8_REF : DataTypeComparator<Byte>(), QuantizedDataType<Byte> {
    
    override val byteSize = 1
    override val priority = 10
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = DT_QUINT8_REF
  }
  
  object QUINT16_REF : DataTypeComparator<Short>(), QuantizedDataType<Short> {
    
    override val byteSize = 2
    override val priority = 50
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = DT_QUINT16_REF
  }
  
  object RESOURCE_REF : DataTypeComparator<Long>(), DataType<Long> {
    
    override val byteSize = -1
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_RESOURCE_REF
  }
  
  object VARIANT_REF : DataTypeComparator<Long>(), DataType<Long> {
    
    override val byteSize = -1
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = DT_VARIANT_REF
  }
}