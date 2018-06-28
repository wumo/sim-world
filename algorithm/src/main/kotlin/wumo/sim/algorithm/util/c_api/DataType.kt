package wumo.sim.algorithm.util.c_api

import org.tensorflow.framework.DataType

fun DataTypefromClass(c: Class<*>): DataType {
  val m = typeCodes
  val a = m[c]
  return typeCodes[c] ?: throw IllegalArgumentException(
      c.name + " objects cannot be used as elements in a TensorFlow Tensor")
}

private val typeCodes = mapOf(
    java.lang.Float::class.java to DataType.DT_FLOAT,
    Float::class.java to DataType.DT_FLOAT,
    java.lang.Double::class.java to DataType.DT_DOUBLE,
    Double::class.java to DataType.DT_DOUBLE,
    Int::class.java to DataType.DT_INT32,
    java.lang.Integer::class.java to DataType.DT_INT32,
    Long::class.java to DataType.DT_INT64,
    java.lang.Long::class.java to DataType.DT_INT64,
    Boolean::class.java to DataType.DT_BOOL,
    java.lang.Boolean::class.java to DataType.DT_BOOL,
    String::class.java to DataType.DT_STRING
)