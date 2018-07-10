package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.DataType

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

fun Int.name(): String {
  return org.tensorflow.framework.DataType.forNumber(this).name.toLowerCase().substring(3)
}

/**Returns a reference `DType` based on this `DType`.*/
val Int.as_ref
  get() = if (is_ref_dytpe) this else this + 100
/**Returns `True` if this `DType` represents a reference type.*/
val Int.is_ref_dytpe
  get() = this > 100
/**Returns a non-reference `DType` based on this `DType`.*/
val Int.base_dtype
  get() = if (is_ref_dytpe) this - 100 else this