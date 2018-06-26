package wumo.sim.algorithm.util.c_api

import org.tensorflow.TensorFlow
import javax.xml.crypto.Data

/** Represents the type of elements in a [Tensor] as an enum.  */
enum class DataType(private val value: Int) {
  /** 32-bit single precision floating point.  */
  FLOAT(1),
  
  /** 64-bit double precision floating point.  */
  DOUBLE(2),
  
  /** 32-bit signed integer.  */
  INT32(3),
  
  /** 8-bit unsigned integer.  */
  UINT8(4),
  
  /**
   * A sequence of bytes.
   *
   *
   * TensorFlow uses the STRING type for an arbitrary sequence of bytes.
   */
  STRING(7),
  
  /** 64-bit signed integer.  */
  INT64(9),
  
  /** Boolean.  */
  BOOL(10);
  
  /** Corresponding value of the TF_DataType enum in the TensorFlow C API.  */
  internal fun c(): Int {
    return value
  }
  
  companion object {
    
    /**
     * Returns the DataType of a Tensor whose elements have the type specified by class `c`.
     *
     * @param c The class describing the TensorFlow type of interest.
     * @return The `DataType` enum corresponding to `c`.
     * @throws IllegalArgumentException if objects of `c` do not correspond to a TensorFlow
     * datatype.
     */
    fun fromClass(c: Class<*>): DataType {
      val m = typeCodes
      val a = m[c]
      return typeCodes[c] ?: throw IllegalArgumentException(
          c.name + " objects cannot be used as elements in a TensorFlow Tensor")
    }
    
    private val values = values()
    
    internal fun fromC(c: Int): DataType {
      for (t in values)
        if (t.value == c)
          return t
      throw IllegalArgumentException(
          "DataType " + c + " is not recognized in Java (version " + TensorFlow.version() + ")")
    }
    
    private val typeCodes = mapOf(
        java.lang.Float::class.java to DataType.FLOAT,
        Float::class.java to DataType.FLOAT,
        java.lang.Double::class.java to DataType.DOUBLE,
        Double::class.java to DataType.DOUBLE,
        Int::class.java to DataType.INT32,
        java.lang.Integer::class.java to DataType.INT32,
        Long::class.java to DataType.INT64,
        java.lang.Long::class.java to DataType.INT64,
        Boolean::class.java to DataType.BOOL,
        java.lang.Boolean::class.java to DataType.BOOL,
        String::class.java to DataType.STRING
    )
  }
}