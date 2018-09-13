package wumo.sim.tensorflow.types

import org.bytedeco.javacpp.BytePointer
import wumo.sim.tensorflow.tf
import wumo.sim.util.NONE
import wumo.sim.util.ndarray.Buf
import java.nio.ByteBuffer

fun Int.toDataType(): DataType<*> =
    DataType.fromCValue(this)

interface DataType<KotlinType> {
  //region Data Type Properties
  val name: String get() = protoType.name
  val cValue: Int get() = protoType.number
  val byteSize: Int
  val priority: Int
  val protoType: org.tensorflow.framework.DataType
  val kotlinType: Class<KotlinType>
  //endregion Data Type Properties
  
  //region Data Type Set Helper Methods
  fun zero(): KotlinType
  
  fun one(): KotlinType
  
  /** Returns `true` if this data type represents a quantized data type. */
  val isQuantized: Boolean
    get() = this in DataType.quantizedDataTypes
  
  /** Returns `true` if this data type represents a non-quantized floating-point data type. */
  val isFloatingPoint: Boolean
    get() = !isQuantized && this in DataType.floatingPointDataTypes
  
  /** Returns `true` if this data type represents a complex data types. */
  val isComplex: Boolean
    get() = this in DataType.complexDataTypes
  
  val isFloatOrComplex: Boolean
    get() = isFloatingPoint || isComplex
  
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
    get() = this == BOOL
  
  ///**Returns a reference `DType` based on this `DType`.*/
  val as_ref: DataType<KotlinType>
    get() = if (isRefType) this else fromCValue(cValue + 100)
  
  ///**Returns `True` if this `DType` represents a reference type.*/
  val isRefType
    get() = cValue > 100
  
  ///**Returns a non-reference `DType` based on this `DType`.*/
  val baseDataType: DataType<KotlinType>
    get() = if (isRefType) fromCValue(cValue - 100) else this
  
  val real: DataType<KotlinType>
    get() = when (baseDataType) {
      is COMPLEX64 -> FLOAT as DataType<KotlinType>
      is COMPLEX128 -> DOUBLE as DataType<KotlinType>
      else -> this
    }
  
  val min: KotlinType get() = NONE()
  val max: KotlinType get() = NONE()
  
  //endregion Data Type Set Helper Methods
  
  fun <R> cast(value: R): KotlinType = TODO()
  
  fun <R> castBuf(value: Buf<R>): Buf<KotlinType> = TODO()
  
  fun put(buffer: BytePointer, idx: Int, element: KotlinType): Unit = TODO()
  
  fun get(buffer: BytePointer, idx: Int): KotlinType = TODO()
  
  override fun equals(that: Any?): Boolean
  
  override fun hashCode(): Int
  override fun toString(): String
  
  companion object {
    //region Data Type Sets
    
    /** Set of all floating-point data types. */
    val floatingPointDataTypes: Set<DataType<*>> = setOf(FLOAT16, FLOAT, DOUBLE, BFLOAT16)
    
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
    private val cvalueMap = mapOf<Int, DataType<*>>(
        BOOL.cValue to BOOL,
        STRING.cValue to STRING,
        FLOAT16.cValue to FLOAT16,
        FLOAT.cValue to FLOAT,
        DOUBLE.cValue to DOUBLE,
        BFLOAT16.cValue to BFLOAT16,
        COMPLEX64.cValue to COMPLEX64,
        COMPLEX128.cValue to COMPLEX128,
        INT8.cValue to INT8,
        INT16.cValue to INT16,
        INT32.cValue to INT32,
        INT64.cValue to INT64,
        UINT8.cValue to UINT8,
        UINT16.cValue to UINT16,
        UINT32.cValue to UINT32,
        UINT64.cValue to UINT64,
        QINT8.cValue to QINT8,
        QINT16.cValue to QINT16,
        QINT32.cValue to QINT32,
        QUINT8.cValue to QUINT8,
        QUINT16.cValue to QUINT16,
        RESOURCE.cValue to RESOURCE,
        VARIANT.cValue to VARIANT,
        
        BOOL_REF.cValue to BOOL_REF,
        STRING_REF.cValue to STRING_REF,
        FLOAT16_REF.cValue to FLOAT16_REF,
        FLOAT_REF.cValue to FLOAT_REF,
        DOUBLE_REF.cValue to DOUBLE_REF,
        BFLOAT16_REF.cValue to BFLOAT16_REF,
        COMPLEX64_REF.cValue to COMPLEX64_REF,
        COMPLEX128_REF.cValue to COMPLEX128_REF,
        INT8_REF.cValue to INT8_REF,
        INT16_REF.cValue to INT16_REF,
        INT32_REF.cValue to INT32_REF,
        INT64_REF.cValue to INT64_REF,
        UINT8_REF.cValue to UINT8_REF,
        UINT16_REF.cValue to UINT16_REF,
        UINT32_REF.cValue to UINT32_REF,
        UINT64_REF.cValue to UINT64_REF,
        QINT8_REF.cValue to QINT8_REF,
        QINT16_REF.cValue to QINT16_REF,
        QINT32_REF.cValue to QINT32_REF,
        QUINT8_REF.cValue to QUINT8_REF,
        QUINT16_REF.cValue to QUINT16_REF,
        RESOURCE_REF.cValue to RESOURCE_REF,
        VARIANT_REF.cValue to VARIANT_REF
    )
    
    fun <D : DataType<*>> fromCValue(cValue: Int) = run {
      val dataType = cvalueMap[cValue] ?: throw IllegalArgumentException(
          "Data type C value '$cValue' is not recognized in Kotlin (TensorFlow version ${tf.version}).")
      dataType as D
    }
    
    /** Returns the data type corresponding to the provided name.
     *
     * @param  name Data type name.
     * @return Data type corresponding to the provided C value.
     * @throws IllegalArgumentException If an invalid data type name is provided.
     */
    private val nameMap: Map<String, DataType<*>> = mapOf(
        BOOL.name to BOOL,
        STRING.name to STRING,
        FLOAT16.name to FLOAT16,
        FLOAT.name to FLOAT,
        DOUBLE.name to DOUBLE,
        BFLOAT16.name to BFLOAT16,
        COMPLEX64.name to COMPLEX64,
        COMPLEX128.name to COMPLEX128,
        INT8.name to INT8,
        INT16.name to INT16,
        INT32.name to INT32,
        INT64.name to INT64,
        UINT8.name to UINT8,
        UINT16.name to UINT16,
        UINT32.name to UINT32,
        UINT64.name to UINT64,
        QINT8.name to QINT8,
        QINT16.name to QINT16,
        QINT32.name to QINT32,
        QUINT8.name to QUINT8,
        QUINT16.name to QUINT16,
        RESOURCE.name to RESOURCE,
        VARIANT.name to VARIANT,
        
        BOOL_REF.name to BOOL_REF,
        STRING_REF.name to STRING_REF,
        FLOAT16_REF.name to FLOAT16_REF,
        FLOAT_REF.name to FLOAT_REF,
        DOUBLE_REF.name to DOUBLE_REF,
        BFLOAT16_REF.name to BFLOAT16_REF,
        COMPLEX64_REF.name to COMPLEX64_REF,
        COMPLEX128_REF.name to COMPLEX128_REF,
        INT8_REF.name to INT8_REF,
        INT16_REF.name to INT16_REF,
        INT32_REF.name to INT32_REF,
        INT64_REF.name to INT64_REF,
        UINT8_REF.name to UINT8_REF,
        UINT16_REF.name to UINT16_REF,
        UINT32_REF.name to UINT32_REF,
        UINT64_REF.name to UINT64_REF,
        QINT8_REF.name to QINT8_REF,
        QINT16_REF.name to QINT16_REF,
        QINT32_REF.name to QINT32_REF,
        QUINT8_REF.name to QUINT8_REF,
        QUINT16_REF.name to QUINT16_REF,
        RESOURCE_REF.name to RESOURCE_REF,
        VARIANT_REF.name to VARIANT_REF
    )
    
    fun <D : DataType<*>> fromName(name: String) = run {
      val dataType = nameMap[name] ?: throw IllegalArgumentException(
          "Data type name '$name' is not recognized in Kotlin (TensorFlow version ${tf.version}).")
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