package wumo.sim.tensorflow.types

import wumo.sim.tensorflow.TF

interface DataType<ScalaType> {
  //region Data Type Properties
  val name: String
  val cValue: Int
  val byteSize: Int
  val priority: Int
  val protoType: org.tensorflow.framework.DataType
  //endregion Data Type Properties
  
  //region Data Type Set Helper Methods
  fun zero(): ScalaType
  
  fun one(): ScalaType
  
  /** Returns `true` if this data type represents a quantized data type. */
  val isQuantized: Boolean
    get() = this in DataType.quantizedDataTypes
  
  /** Returns `true` if this data type represents a non-quantized floating-point data type. */
  val isFloatingPoint: Boolean
    get() = !isQuantized && this in DataType.floatingPointDataTypes
  
  /** Returns `true` if this data type represents a complex data types. */
  val isComplex: Boolean
    get() = this in DataType.complexDataTypes
  
  /** Returns `true` if this data type represents a non-quantized integer data type. */
  val isInteger: Boolean
    get() = !isQuantized && this in DataType.integerDataTypes
  
  /** Returns `true` if this data type represents a non-quantized unsigned data type. */
  val isUnsigned: Boolean
    get() = !isQuantized && this in DataType.unsignedDataTypes
  
  /** Returns `true` if this data type represents a numeric data type. */
  val isNumeric: Boolean
    get() = this in DataType.numericDataTypes
  
  /** Returns `true` if this data type represents a boolean data type. */
  val isBoolean: Boolean
    get() = this == BOOLEAN
  
  //endregion Data Type Set Helper Methods
  
  override fun equals(that: Any?): Boolean
  
  override fun hashCode(): Int
  override fun toString(): String
  
  companion object {
    //region Data Type Sets
    
    /** Set of all floating-point data types. */
    val floatingPointDataTypes: Set<DataType<*>> = setOf(FLOAT16, FLOAT32, FLOAT64, BFLOAT16)
    
    /** Set of all complex data types. */
    val complexDataTypes: Set<DataType<*>> = setOf(COMPLEX64, COMPLEX128)
    
    /** Set of all integer data types. */
    val integerDataTypes: Set<DataType<*>> = setOf(INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, QINT8, QINT16, QINT32, QUINT8, QUINT16)
    
    /** Set of all quantized data types. */
    val quantizedDataTypes: Set<DataType<*>> = setOf(BFLOAT16, QINT8, QINT16, QINT32, QUINT8, QUINT16)
    
    /** Set of all unsigned data types. */
    val unsignedDataTypes: Set<DataType<*>> = setOf(UINT8, UINT16, UINT32, UINT64, QUINT8, QUINT16)
    
    /** Set of all numeric data types. */
    val numericDataTypes: Set<DataType<*>> = floatingPointDataTypes + complexDataTypes + integerDataTypes + quantizedDataTypes
    
    //endregion Data Type Sets
    
    //region Helper Methods
    
    /** Returns the [DataType] of the provided value.
     *
     * @param  value Value whose data type to return.
     * @return Data type of the provided value.
     */
    fun <T, D : DataType<*>> dataTypeOf(value: T, evSupported: SupportedType<T, D>) = evSupported.dataType
    
    /** Returns the data type corresponding to the provided C value.
     *
     * By C value here we refer to an integer representing a data type in the `TF_DataType` enum of the TensorFlow C
     * API.
     *
     * @param  cValue C value.
     * @return Data type corresponding to the provided C value.
     * @throws IllegalArgumentException If an invalid C value is provided.
     */
    fun <D : DataType<*>> fromCValue(cValue: Int) = run {
      val dataType = when (cValue) {
        BOOLEAN.cValue -> BOOLEAN
        STRING.cValue -> STRING
        FLOAT16.cValue -> FLOAT16
        FLOAT32.cValue -> FLOAT32
        FLOAT64.cValue -> FLOAT64
        BFLOAT16.cValue -> BFLOAT16
        COMPLEX64.cValue -> COMPLEX64
        COMPLEX128.cValue -> COMPLEX128
        INT8.cValue -> INT8
        INT16.cValue -> INT16
        INT32.cValue -> INT32
        INT64.cValue -> INT64
        UINT8.cValue -> UINT8
        UINT16.cValue -> UINT16
        UINT32.cValue -> UINT32
        UINT64.cValue -> UINT64
        QINT8.cValue -> QINT8
        QINT16.cValue -> QINT16
        QINT32.cValue -> QINT32
        QUINT8.cValue -> QUINT8
        QUINT16.cValue -> QUINT16
        RESOURCE.cValue -> RESOURCE
        VARIANT.cValue -> VARIANT
        else -> throw IllegalArgumentException(
            "Data type C value '$cValue' is not recognized in Scala (TensorFlow version ${TF.version}).")
      }
      dataType as D
    }
    
    /** Returns the data type corresponding to the provided name.
     *
     * @param  name Data type name.
     * @return Data type corresponding to the provided C value.
     * @throws IllegalArgumentException If an invalid data type name is provided.
     */
    fun <D : DataType<*>> fromName(name: String) = run {
      val dataType = when (name) {
        "BOOLEAN" -> BOOLEAN
        "STRING" -> STRING
        "FLOAT16" -> FLOAT16
        "FLOAT32" -> FLOAT32
        "FLOAT64" -> FLOAT64
        "BFLOAT16" -> BFLOAT16
        "COMPLEX64" -> COMPLEX64
        "COMPLEX128" -> COMPLEX128
        "INT8" -> INT8
        "INT16" -> INT16
        "INT32" -> INT32
        "INT64" -> INT64
        "UINT8" -> UINT8
        "UINT16" -> UINT16
        "UINT32" -> UINT32
        "UINT64" -> UINT64
        "QINT8" -> QINT8
        "QINT16" -> QINT16
        "QINT32" -> QINT32
        "QUINT8" -> QUINT8
        "QUINT16" -> QUINT16
        "RESOURCE" -> RESOURCE
        "VARIANT" -> VARIANT
        else -> throw IllegalArgumentException(
            "Data type name '$name' is not recognized in Scala (TensorFlow version ${TF.version}).")
      }
      dataType as D
    }
    
    /** Returns the most precise data type out of the provided data types, based on their `priority` field.
     *
     * @param  dataTypes Data types out of which to pick the most precise.
     * @return Most precise data type in `dataTypes`.
     */
    fun mostPrecise(vararg dataTypes: DataType<*>): DataType<*> = dataTypes.maxBy { it.priority }!!
    
    /** Returns the most precise data type out of the provided data types, based on their `priority` field.
     *
     * @param  dataTypes Data types out of which to pick the most precise.
     * @return Most precise data type in `dataTypes`.
     */
    fun leastPrecise(vararg dataTypes: DataType<*>): DataType<*> = dataTypes.minBy { it.priority }!!
    
    //endregion Helper Methods
    
  }
}