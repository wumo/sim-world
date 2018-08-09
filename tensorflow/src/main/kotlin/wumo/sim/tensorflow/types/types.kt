package wumo.sim.tensorflow.types

import wumo.sim.util.NOPE

typealias STRING = types.STRING
typealias BOOLEAN = types.BOOLEAN
typealias FLOAT16 = types.FLOAT16
typealias FLOAT32 = types.FLOAT32
typealias FLOAT64 = types.FLOAT64
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
  
  object STRING : ReducibleDataType<String> {
    override val name = "STRING"
    override val cValue = 7
    override val byteSize = -1
    override val priority = 1000
    
    override fun zero() = ""
    override fun one() = NOPE()
    
    override val protoType = org.tensorflow.framework.DataType.DT_STRING
    
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object BOOLEAN : ReducibleDataType<Boolean> {
    override val name = "BOOLEAN"
    override val cValue = 10
    override val byteSize = 1
    override val priority = 0
    
    override fun zero() = false
    override fun one() = true
    
    override val protoType = org.tensorflow.framework.DataType.DT_BOOL
    
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object FLOAT16 : Float16OrFloat32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    override val name = "FLOAT16"
    override val cValue = 19
    override val byteSize = 2
    override val priority = -1
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = org.tensorflow.framework.DataType.DT_HALF
    
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object FLOAT32 : Float32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    override val name = "FLOAT32"
    override val cValue = 1
    override val byteSize = 4
    override val priority = 220
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = org.tensorflow.framework.DataType.DT_FLOAT
    
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object FLOAT64 : Float32OrFloat64<Double> {
    override val name = "FLOAT64"
    override val cValue = 2
    override val byteSize = 8
    override val priority = 230
    
    override fun zero() = 0.0
    override fun one() = 1.0
    
    override val protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_DOUBLE
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object BFLOAT16 : BFloat16OrFloat32OrFloat64<Float>, BFloat16OrFloat16OrFloat32<Float> {
    override val name = "BFLOAT16"
    override val cValue = 14
    override val byteSize = 2
    override val priority = -1
    
    override fun zero() = 0.0f
    override fun one() = 1.0f
    
    override val protoType = org.tensorflow.framework.DataType.DT_BFLOAT16
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object COMPLEX64 : ComplexDataType<Double> {
    override val name = "COMPLEX64"
    override val cValue = 8
    override val byteSize = 8
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = org.tensorflow.framework.DataType.DT_COMPLEX64
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object COMPLEX128 : ComplexDataType<Double> {
    override val name = "COMPLEX128"
    override val cValue = 18
    override val byteSize = 16
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = org.tensorflow.framework.DataType.DT_COMPLEX128
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object INT8 : IntOrUInt<Byte> {
    override val name = "INT8"
    override val cValue = 6
    override val byteSize = 1
    override val priority = 40
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_INT8
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object INT16 : IntOrUInt<Short> {
    override val name = "INT16"
    override val cValue = 5
    override val byteSize = 2
    override val priority = 80
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_INT16
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object INT32 : Int32OrInt64<Int> {
    override val name = "INT32"
    override val cValue = 3
    override val byteSize = 4
    override val priority = 100
    
    override fun zero() = 0
    override fun one() = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_INT32
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object INT64 : Int32OrInt64<Long> {
    override val name = "INT64"
    override val cValue = 9
    override val byteSize = 8
    override val priority = 110
    
    override fun zero() = 0L
    override fun one() = 1L
    
    override val protoType = org.tensorflow.framework.DataType.DT_INT64
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object UINT8 : UInt8OrInt32OrInt64<Byte> {
    override val name = "UINT8"
    override val cValue = 4
    override val byteSize = 1
    override val priority = 20
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_UINT8
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object UINT16 : IntOrUInt<Short> {
    override val name = "UINT16"
    override val cValue = 17
    override val byteSize = 2
    override val priority = 60
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_UINT16
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object UINT32 : IntOrUInt<Long> {
    override val name = "UINT32"
    override val cValue = 22
    override val byteSize = 4
    override val priority = 85
    
    override fun zero(): Long = 0
    override fun one(): Long = 1
    
    override val protoType: org.tensorflow.framework.DataType = NOPE()
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object UINT64 : IntOrUInt<Long> {
    override val name = "UINT64"
    override val cValue = 23
    override val byteSize = 8
    override val priority = 105
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = NOPE()
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object QINT8 : QuantizedDataType<Byte> {
    override val name = "QINT8"
    override val cValue = 11
    override val byteSize = 1
    override val priority = 30
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_QINT8
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object QINT16 : QuantizedDataType<Short> {
    override val name = "QINT16"
    override val cValue = 15
    override val byteSize = 2
    override val priority = 70
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_QINT16
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object QINT32 : QuantizedDataType<Int> {
    override val name = "QINT32"
    override val cValue = 13
    override val byteSize = 4
    override val priority = 90
    
    override fun zero(): Int = 0
    override fun one(): Int = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_QINT32
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object QUINT8 : QuantizedDataType<Byte> {
    override val name = "QUINT8"
    override val cValue = 12
    override val byteSize = 1
    override val priority = 10
    
    override fun zero(): Byte = 0
    override fun one(): Byte = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_QUINT8
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object QUINT16 : QuantizedDataType<Short> {
    override val name = "QUINT16"
    override val cValue = 16
    override val byteSize = 2
    override val priority = 50
    
    override fun zero(): Short = 0
    override fun one(): Short = 1
    
    override val protoType = org.tensorflow.framework.DataType.DT_QUINT16
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object RESOURCE : DataType<Long> {
    override val name = "RESOURCE"
    override val cValue = 20
    override val byteSize = -1
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = org.tensorflow.framework.DataType.DT_RESOURCE
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
  
  object VARIANT : DataType<Long> {
    override val name = "VARIANT"
    override val cValue = 21
    override val byteSize = -1
    override val priority = -1
    
    override fun zero() = NOPE()
    override fun one() = NOPE()
    
    override val protoType = org.tensorflow.framework.DataType.DT_VARIANT
    override fun toString() = name
    override fun equals(that: Any?) =
        if (that is DataType<*>) cValue == that.cValue
        else false
    
    override fun hashCode() = cValue
  }
}