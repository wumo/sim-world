package wumo.sim.tensorflow

import org.bytedeco.javacpp.tensorflow.*

private val typeCodes = mapOf(
    java.lang.Float::class.java to DT_FLOAT,
    Float::class.java to DT_FLOAT,
    java.lang.Double::class.java to DT_DOUBLE,
    Double::class.java to DT_DOUBLE,
    Boolean::class.java to DT_BOOL,
    java.lang.Boolean::class.java to DT_BOOL,
    Byte::class.java to DT_INT8,
    java.lang.Byte::class.java to DT_INT8,
    Short::class.java to DT_INT16,
    java.lang.Short::class.java to DT_INT16,
    Int::class.java to DT_INT32,
    java.lang.Integer::class.java to DT_INT32,
    Long::class.java to DT_INT64,
    java.lang.Long::class.java to DT_INT64,
    String::class.java to DT_STRING
)

fun dtypeFromClass(c: Class<*>): Int {
  return typeCodes[c] ?: throw IllegalArgumentException("${c.name} objects cannot be used as elements in a TensorFlow Output")
}

fun dtypeToClass(dtype: Int): Class<*> {
  return when (dtype) {
    DT_INT8, DT_UINT8 -> Byte::class.java
    DT_FLOAT -> Float::class.java
    DT_DOUBLE -> Double::class.java
    DT_BOOL -> Boolean::class.java
    DT_INT16, DT_UINT16 -> Short::class.java
    DT_INT32, DT_UINT32 -> Int::class.java
    DT_INT64, DT_UINT64 -> Long::class.java
    DT_STRING -> String::class.java
    else -> throw IllegalArgumentException("$dtype not supported")
  }
}

fun Int.name(): String {
  return org.tensorflow.framework.DataType.forNumber(this).name.toLowerCase().substring(3)
}

fun Int.orUse(dtype: Int) = if (this <= 0) dtype else this

///**Returns a reference `DType` based on this `DType`.*/
//val Int.as_ref
//  get() = if (is_ref_dytpe) this else this + 100
///**Returns `True` if this `DType` represents a reference type.*/
//val Int.is_ref_dytpe
//  get() = this > 100
///**Returns a non-reference `DType` based on this `DType`.*/
//val Int.base_dtype
//  get() = if (is_ref_dytpe) this - 100 else this
//
//val int_type = setOf(DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64)
//val Int.is_integer: Boolean
//  get() = !is_quantized && base_dtype in int_type
//
//val quantized_type = setOf(DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32)
//val Int.is_quantized
//  get() = this.base_dtype in quantized_type