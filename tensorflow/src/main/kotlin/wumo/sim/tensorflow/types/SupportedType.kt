package wumo.sim.tensorflow.types

import wumo.sim.tensorflow.core.InvalidDataTypeException

sealed class SupportedType<out T, D : DataType<*>> {
  abstract val dataType: D
  open fun cast(value: Any): T = throw InvalidDataTypeException("The Kotlin type of this data type is not supported.")
}

object stringIsSupported : SupportedType<String, STRING>() {
  override val dataType = STRING
  override fun cast(value: Any) = value.toString()
}

object booleanIsSupported : SupportedType<Boolean, BOOLEAN>() {
  override val dataType = BOOLEAN
  override fun cast(value: Any): Boolean = when (value) {
    is Boolean -> value
    else -> throw InvalidDataTypeException("Cannot convert the provided value a boolean.")
  }
}

object floatIsSupported : SupportedType<Float, FLOAT32>() {
  override val dataType = FLOAT32
  override fun cast(value: Any): Float = when (value) {
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

object doubleIsSupported : SupportedType<Double, FLOAT64>() {
  override val dataType = FLOAT64
  override fun cast(value: Any): Double = when (value) {
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

object byteIsSupported : SupportedType<Byte, INT8>() {
  override val dataType = INT8
  override fun cast(value: Any): Byte = when (value) {
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

object shortIsSupported : SupportedType<Short, INT16>() {
  override val dataType = INT16
  override fun cast(value: Any): Short = when (value) {
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

object intIsSupported : SupportedType<Int, INT32>() {
  override val dataType = INT32
  override fun cast(value: Any): Int = when (value) {
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

object longIsSupported : SupportedType<Long, INT64>() {
  override val dataType = INT64
  override fun cast(value: Any): Long = when (value) {
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

object uByteIsSupported : SupportedType<Byte, UINT8>() {
  override val dataType = UINT8
  override fun cast(value: Any): Byte = when (value) {
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

object uShortIsSupported : SupportedType<Short, UINT16>() {
  override val dataType = UINT16
  override fun cast(value: Any): Short = when (value) {
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