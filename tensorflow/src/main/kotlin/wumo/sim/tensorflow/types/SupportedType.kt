package wumo.sim.tensorflow.types

import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.util.SwitchOnClass

val switchKotlinType = SwitchOnClass<DataType<*>>().apply {
  case<String> { STRING }
  case<Boolean> { BOOL }
  case<Float> { FLOAT }
  case<Double> { DOUBLE }
  case<Byte> { INT8 }
  case<Short> { INT16 }
  case<Int> { INT32 }
  case<Long> { INT64 }
}

fun <T : Any> Class<T>.toDataType(): DataType<T> =
    switchKotlinType(this) as DataType<T>

val Int.kotlinType: Class<*>
  get() {
    return when (this) {
      COMPLEX64.cValue, FLOAT.cValue -> Float::class.java
      DOUBLE.cValue -> Double::class.java
      BOOL.cValue -> Boolean::class.java
      QUINT8.cValue, UINT8.cValue, QINT8.cValue, INT8.cValue -> Byte::class.java
      BFLOAT16.cValue, INT16.cValue, UINT16.cValue -> Short::class.java
      QINT32.cValue, INT32.cValue -> Int::class.java
      INT64.cValue, UINT64.cValue -> Long::class.java
      STRING.cValue -> String::class.java
      else -> throw IllegalArgumentException("$this not supported")
    }
  }

fun Int.name(): String {
  return org.tensorflow.framework.DataType.forNumber(this).name.toLowerCase().substring(3)
}

val supportedTypes = mutableMapOf<DataType<*>, SupportedType<*, *>>()

sealed class SupportedType<out T, D : DataType<*>>(val dataType: D) {
  open fun <R> cast(value: R): T = throw InvalidDataTypeException("The Kotlin type of this data type is not supported.")
  
  init {
    supportedTypes[dataType] = this
  }
}

object stringIsSupported : SupportedType<String, types.STRING>(STRING) {
  override fun <R> cast(value: R) = value.toString()
}

object booleanIsSupported : SupportedType<Boolean, types.BOOL>(BOOL) {
  override fun <R> cast(value: R): Boolean = when (value) {
    is Boolean -> value
    else -> throw InvalidDataTypeException("Cannot convert the provided value a boolean.")
  }
}

object floatIsSupported : SupportedType<Float, types.FLOAT32>(FLOAT) {
  override fun <R> cast(value: R): Float = when (value) {
    is Boolean -> if (value) 1.0f else 0.0f
    is Float -> value.toFloat()
    is Double -> value.toFloat()
    is Byte -> value.toFloat()
    is Short -> value.toFloat()
    is Int -> value.toFloat()
    is Long -> value.toFloat()
    else -> throw InvalidDataTypeException("Cannot convert the provided value a float.")
  }
}

object doubleIsSupported : SupportedType<Double, types.FLOAT64>(DOUBLE) {
  override fun <R> cast(value: R): Double = when (value) {
    is Boolean -> if (value) 1.0 else 0.0
    is Float -> value.toDouble()
    is Double -> value.toDouble()
    is Byte -> value.toDouble()
    is Short -> value.toDouble()
    is Int -> value.toDouble()
    is Long -> value.toDouble()
    else -> throw InvalidDataTypeException("Cannot convert the provided value a double.")
  }
}

object byteIsSupported : SupportedType<Byte, types.INT8>(INT8) {
  override fun <R> cast(value: R): Byte = when (value) {
    is Boolean -> if (value) 1 else 0
    is Float -> value.toByte()
    is Double -> value.toByte()
    is Byte -> value.toByte()
    is Short -> value.toByte()
    is Int -> value.toByte()
    is Long -> value.toByte()
    else -> throw InvalidDataTypeException("Cannot convert the provided value a byte.")
  }
}

object shortIsSupported : SupportedType<Short, types.INT16>(INT16) {
  override fun <R> cast(value: R): Short = when (value) {
    is Boolean -> if (value) 1 else 0
    is Float -> value.toShort()
    is Double -> value.toShort()
    is Byte -> value.toShort()
    is Short -> value.toShort()
    is Int -> value.toShort()
    is Long -> value.toShort()
    else -> throw InvalidDataTypeException("Cannot convert the provided value a short.")
  }
}

object intIsSupported : SupportedType<Int, types.INT32>(INT32) {
  override fun <R> cast(value: R): Int = when (value) {
    is Boolean -> if (value) 1 else 0
    is Float -> value.toInt()
    is Double -> value.toInt()
    is Byte -> value.toInt()
    is Short -> value.toInt()
    is Int -> value.toInt()
    is Long -> value.toInt()
    else -> throw InvalidDataTypeException("Cannot convert the provided value an integer.")
  }
}

object longIsSupported : SupportedType<Long, types.INT64>(INT64) {
  override fun <R> cast(value: R): Long = when (value) {
    is Boolean -> if (value) 1L else 0L
    is Float -> value.toLong()
    is Double -> value.toLong()
    is Byte -> value.toLong()
    is Short -> value.toLong()
    is Int -> value.toLong()
    is Long -> value.toLong()
    else -> throw InvalidDataTypeException("Cannot convert the provided value a long.")
  }
}

object uByteIsSupported : SupportedType<Byte, types.UINT8>(UINT8) {
  override fun <R> cast(value: R): Byte = when (value) {
    is Boolean -> if (value) 1 else 0
    is Float -> value.toByte()
    is Double -> value.toByte()
    is Byte -> value
    is Short -> value.toByte()
    is Int -> value.toByte()
    is Long -> value.toByte()
    else -> throw InvalidDataTypeException("Cannot convert the provided value an unsigned byte.")
  }
}

object uShortIsSupported : SupportedType<Short, types.UINT16>(UINT16) {
  override fun <R> cast(value: R): Short = when (value) {
    is Boolean -> if (value) 1 else 0
    is Float -> value.toShort()
    is Double -> value.toShort()
    is Byte -> value.toShort()
    is Short -> value.toShort()
    is Int -> value.toShort()
    is Long -> value.toShort()
    else -> throw InvalidDataTypeException("Cannot convert the provided value an unsigned short.")
  }
}