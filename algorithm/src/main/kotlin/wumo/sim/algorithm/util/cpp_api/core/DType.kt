package wumo.sim.algorithm.util.cpp_api.core

import org.bytedeco.javacpp.tensorflow.*

private val typeCodes = mapOf(
    java.lang.Float::class.java to DT_FLOAT,
    Float::class.java to DT_FLOAT,
    java.lang.Double::class.java to DT_DOUBLE,
    Double::class.java to DT_DOUBLE,
    Int::class.java to DT_INT32,
    java.lang.Integer::class.java to DT_INT32,
    Long::class.java to DT_INT64,
    java.lang.Long::class.java to DT_INT64,
    Boolean::class.java to DT_BOOL,
    java.lang.Boolean::class.java to DT_BOOL,
    Byte::class.java to DT_INT8,
    java.lang.Byte::class.java to DT_INT8,
    String::class.java to DT_STRING
)

fun dtypeFromClass(c: Class<*>): Int {
  return typeCodes[c] ?: throw IllegalArgumentException("${c.name} objects cannot be used as elements in a TensorFlow Tensor")
}