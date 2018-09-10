package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.core.InvalidShapeException
import wumo.sim.tensorflow.ops.basic.get
import wumo.sim.tensorflow.tensor.constantValue
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.INT64
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.NDArray

/** Class wrapping dynamic-sized, per-time-step, write-once tensor arrays.
 *
 * This class is meant to be used with dynamic iteration primitives such as `whileLoop` and `mapFunction`. It supports
 * gradient back-propagation via special "flow" control flow dependencies.
 *
 * Note that the name of the `TensorArray` (even if passed in) is uniquified automatically. Each time a new
 * `TensorArray` is created at runtime it is assigned its own name for the duration of the run. This avoids name
 * collisions if a `TensorArray` is created within a `whileLoop`.
 *
 * @param  handle                 Tensor handle to the tensor array.
 * @param  flow                   Float scalar tensor for the tensor array, used to control gradient flow.
 * @param  dataType               Data type of the tensor array elements.
 * @param  inferShape             Boolean value indicating whether shape inference is enabled. If `true`, all elements
 *                                must have the same shape.
 * @param  elementShape           A [[Shape]] object specifying the shape constraints of each of the elements of the
 *                                tensor array. The shape need not be fully defined.
 * @param  colocateWithFirstWriteCall Boolean value indicating whether to place the tensor array on the same device as
 *                                the tensor used on its first write call (write operations include `write`,
 *                                `unstack`, and `split`). If `false`, the tensor array will be placed on the device
 *                                determined by the op creation context available during its initialization.
 * @param  colocationOps          Used to keep track of what ops the tensor array should be colocated with.
 *
 * @author Emmanouil Antonios Platanios
 */
class TensorArray private constructor(
    val handle: Output,
    val flow: Output,
    val dataType: DataType<*>,
    val inferShape: Boolean,
    var elementShape: Shape?,
    val colocateWithFirstWriteCall: Boolean = true,
    var colocationOps: List<Op>? = null) : OutputConvertible {
  
  /** Returns a tensor array with the same content and properties as this one.
   *
   * @return New [TensorArray] object with a flow that ensures the control dependencies from the contexts will become
   *         control dependencies for writes, reads, etc. Use this object for all subsequent operations.
   */
  val identity: TensorArray
    get() = TensorArray(handle, tf.identity(flow), dataType, inferShape,
                        elementShape, colocateWithFirstWriteCall, colocationOps)
  
  /** Creates an op that reads an element from this tensor array.
   *
   * @param  index Position to read from, inside the tensor array.
   * @param  name  Name for the created op.
   * @return Tensor in the specified position of the tensor array.
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#read"
   */
  fun read(index: Output, name: String = "TensorArrayRead"): Output =
      tf.colocateWith(mutableSetOf(handle.op)) {
        val value = tf.tensorArrayReadV3(handle, index, flow, dataType, name)
        elementShape?.let { value.setShape(it) }
        value
      }
  
  /** Creates an op that writes an element to this tensor array.
   *
   * @param  index Position to write to, inside the tensor array.
   * @param  value Tensor to write to the tensor array.
   * @param  name  Name for the created op.
   * @return Output flow of the tensor array, used to enforce proper chaining of operations.
   */
  fun write(index: Output, value: Output, name: String = "TensorArrayWrite"): TensorArray {
    val writeFlow = maybeColocateWith(value.op) {
      tf.tensorArrayWriteV3(handle, index, value, flow, name)
    }
    val returnValue = TensorArray(
        handle, writeFlow, dataType, inferShape, elementShape,
        colocateWithFirstWriteCall, colocationOps)
    if (inferShape)
      returnValue.mergeElementShape(value.shape)
    return returnValue
  }
  
  /** Creates an op that gathers specific elements from this tensor array.
   *
   * Note that all elements selected by `indices` must have been written and must have the same shape.
   *
   * @param  indices One-dimensional tensor containing the positions in the tensor array from which to read tensor
   *                 elements.
   * @param  name    Name for the created op.
   * @return Tensor containing the gathered elements, concatenated along a new axis (the new dimension `0`).
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#gather"
   */
  fun gather(indices: Output, name: String = "TensorArrayGather"): Output =
      tf.colocateWith(handle.op) {
        val ind = if (indices.rank == 0) tf.expandDims(indices, tf.const(0)) else indices
        val value = tf.tensorArrayGatherV3(handle, ind, flow, dataType,
                                           elementShape ?: Shape(), name)
        elementShape?.let {
          value.setShape(Shape(intArrayOf(-1, *it.asIntArray()!!)))
        }
        value
      }
  
  /** Creates an op that returns the elements in this tensor array as a stacked tensor.
   *
   * Note that all elements of this tensor array must have been written and must have the same shape.
   *
   * If the elements have rank `R`, then the returned tensor shape will be equal to `R + 1`.
   *
   * @param  name Name for the created op.
   * @return Stacked tensor.
   */
  fun stack(name: String = "TensorArrayStack"): Output =
      tf.nameScope(name, setOf(handle.op)) {
        tf.colocateWith(handle.op) {
          gather(tf.range(tf.const(0), size()), name)
        }
      }
  
  /** Creates an op that scatters the provided elements along indices of this tensor array.
   *
   * Note that `indices` must be a vector and its length must match the first dimension of `value`.
   *
   * @param  indices One-dimensional tensor containing the positions in the tensor array at which to write the tensor
   *                 elements.
   * @param  value   Concatenated tensor to write to the tensor array.
   * @param  name    Name for the created op.
   * @return Output flow of the tensor array, used to enforce proper chaining of operations.
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#scatter"
   */
  fun scatter(indices: Output, value: Output, name: String = "TensorArrayScatter"): TensorArray =
      tf.nameScope("TensorArrayScatter") {
        val scatterFlow = maybeColocateWith(value.op) {
          tf.tensorArrayScatterV3(handle, indices, value, flow, tf.currentNameScope)
        }
        val returnValue = TensorArray(
            handle, scatterFlow, dataType, inferShape, elementShape,
            colocateWithFirstWriteCall, colocationOps)
        if (inferShape) {
          val valueShape = scatterFlow.op.inputs[2].shape
          val shape = if (valueShape.isUnknown) valueShape
          else valueShape.slice(1)
          returnValue.mergeElementShape(shape)
        }
        returnValue
      }
  
  /** Creates an op that unstacks the values of a tensor in this tensor array.
   *
   * If the input value shapes have rank `R`, then the output tensor array will contain elements whose shapes have
   * rank `R - 1`.
   *
   * @param  value Tensor to unstack.
   * @param  name  Name for the created op.
   * @return New tensor array object with flow that ensures the unstack occurs. Use this object for all subsequent
   *         operations.
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#unstack"
   */
  fun unstack(value: Output, name: String = "TensorArrayUnstack"): TensorArray =
      tf.nameScope("TensorArrayUnstack") {
        scatter(tf.range(tf.const(0), tf.shape(value)[0]), value,
                tf.currentNameScope)
      }
  
  /** Creates an op that concatenates the elements of the tensor array.
   *
   * The op takes `T` elements with shapes `[n0, d0, d1, ...]`, `[n1, d0, d1, ...]`, ..., `[n(T-1), d0, d1, ...]` and
   * concatenates them into a tensor with shape `[n0 + n1 + ... + n(T-1), d0, d1, ...]`.
   *
   * All elements must have been written and must have the same shape, except for their first dimension.
   *
   * @param  name Name for the created op.
   * @return Tensor with all of the elements in the tensor array, concatenated along the first axis.
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#concat"
   */
  fun concat(name: String = "TensorArrayConcatenate"): Output {
    val shape = elementShape?.slice(1) ?: Shape()
    val (value, _) = tf.tensorArrayConcatV3(handle, flow, dataType, shape, name)
    elementShape?.let {
      value.setShape(Shape(intArrayOf(-1, *it.slice(1).asIntArray()!!)))
    }
    return value
  }
  
  /** Splits the values of a tensor into a tensor array.
   *
   * @param  input   (N+1)-dimensional tensor to split. Must have the same data type as this tensor array.
   * @param  lengths 1-D integer tensor with the lengths to use when splitting `input` along its first dimension.
   * @param  name    Name for the created op.
   * @return Tensor array with flow that ensures the split occurs. Use this object for all subsequent operations.
   * @see ""
   */
  fun split(input: Output, lengths: Output, name: String = "TensorArraySplit"): TensorArray =
      tf.nameScope(name, setOf(handle.op, input.op, lengths.op)) {
        val splitFlow = maybeColocateWith(input.op) {
          tf.tensorArraySplitV3(handle, input, tf.cast(lengths, INT64), flow, name)
        }
        val returnValue = TensorArray(
            handle, splitFlow, dataType, inferShape, elementShape,
            colocateWithFirstWriteCall, colocationOps)
        if (inferShape) {
          val valueShape = splitFlow.op.inputs[1].shape
          val clengths = constantValue(splitFlow.op.inputs[2])
          clengths as NDArray<Int>?
          val shape = if (valueShape.rank != -1 && clengths != null && clengths.max() == clengths.min())
            Shape(intArrayOf((clengths.get() as Long).toInt(), *valueShape.slice(1).asIntArray()!!))
          else
            Shape()
          returnValue.mergeElementShape(shape)
        }
        returnValue
      }
  
  /** Changes the element shape of the array given a shape to merge with.
   *
   * @param  shape Shape to merge with.
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#_merge_element_shape"
   */
  private fun mergeElementShape(shape: Shape) {
    elementShape?.let {
      if (!shape.isCompatibleWith(it))
        throw InvalidShapeException("Expected shape '$it' but got '$shape' (and inferShape = true).")
      elementShape = it.mergeWith(shape)
    } ?: run {
      if (shape.rank != -1)
        elementShape = shape
    }
  }
  
  /** Returns an op that gets the current size of the tensor array.
   *
   * @param  name Name for the created op.
   * @return Created op output, containing the current size of the tensor array.
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#size"
   */
  fun size(name: String = "TensorArraySize"): Output =
      tf.colocateWith(handle.op) {
        tf.tensorArraySizeV3(handle, flow, name)
      }
  
  /** Colocates ops created by `block` with an internal colocation group, if such a group exists, or with `op`. If no
   * internal colocation group is set, this method colocates ops with `op` and sets the internal colocation group to be
   * `op`.
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#_maybe_colocate_with"
   */
  private fun <R> maybeColocateWith(op: Op, block: () -> R): R =
      if (!colocateWithFirstWriteCall)
        block()
      else if (colocationOps == null) {
        colocationOps = listOf(op)
        tf.colocateWith(colocationOps!!.toMutableSet()) {
          block()
        }
      } else
        tf.colocateWith(colocationOps!![0]) {
          block()
        }
  
  /** Returns a tensor array for storing the gradients of the values stored in this tensor array.
   *
   * If the provided tensor array gradient already exists, then a reference to it is returned.
   *
   * This op locks the size of the original tensor array by disabling its dynamic size flag.
   *
   * ==A Note About the Input `flow`==
   *
   * The handle `flow` forces the execution of the gradient lookup to occur only after certain other operations have
   * occurred. For example, when the forward tensor array is dynamically sized, writes to this tensor array may resize
   * the object. The gradient tensor array is statically sized based on the size of the forward tensor array when this
   * operation executes. Furthermore, the size of the forward tensor array is frozen by this call. As a result, the
   * flow is used to ensure that the call to generate the gradient tensor array only happens after all writes are
   * executed.
   *
   * In the case of dynamically sized tensor arrays, the gradient computation should only be performed on read
   * operations that have themselves been chained via flow to occur only after all writes have executed. That way the
   * final size of the forward tensor array is known when this operation is called.
   *
   * ==A Note About the `source` Attribute==
   *
   * Tensor array gradient calls use an accumulator tensor array object. If multiple gradients are calculated and run
   * in the same session, then the multiple gradient nodes may accidentally flow though the same accumulator tensor
   * array. This double counts and generally breaks the tensor array gradient flow.
   *
   * The solution is to identify which gradient call this particular tensor array gradient is being called from. This
   * is performed by identifying a unique string (e.g. "gradients", "gradients_1", ...) from the input gradient
   * tensor's name. This string is used as a suffix when creating the tensor array gradient object here (the attribute
   * `source`).
   *
   * The attribute `source` is added as a suffix to the forward tensor array's name when performing the
   * creation/lookup, so that each separate gradient calculation gets its own tensor array accumulator.
   *
   * @param  source Gradient source string used to decide which gradient tensor array to return.
   * @param  flow   Float scalar that enforces proper chaining of operations.
   * @param  name   Name for the created gradient op.
   * @return Gradient tensor array.
   * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#grad"
   */
  fun gradient(source: String,
               flow: Output = this.flow,
               name: String = "TensorArrayGrad"
  ): TensorArray =
  // `TensorArray.gradientOp` requires a flow input when forward tensor arrays are dynamically sized. This forces the
  // creation of the gradient tensor array only once the final forward array's size is fixed.
      tf.nameScope(name, setOf(handle.op)) {
        tf.colocateWith(handle.op) {
          val (gradientHandle, _) = tf.tensorArrayGradV3(handle, flow, source)
          val gradientFlow = tf.controlDependencies(gradientHandle.op) {
            tf.identity(flow, name = "gradient_flow")
          }
          TensorArray(
              gradientHandle, gradientFlow, dataType, inferShape, elementShape, colocateWithFirstWriteCall = false)
        }
      }
  
  /** Converts this tensor array to an output (i.e., dense symbolic tensor), by stacking it. */
  override fun toOutput(): Output = stack()
  
  companion object {
    /** Creates a new tensor array.
     *
     * @param  size                   Size of the tensor array.
     * @param  dataType               Data type of the elements in the tensor array.
     * @param  dynamicSize            Boolean value indicating whether writes to the tensor array are allowed to grow in
     *                                size. By default, this is not allowed.
     * @param  clearAfterRead         Boolean value indicating whether to clear the tensors in the array, after being
     *                                read. This disables multiple read semantics but allows early release of memory.
     *                                Defaults to `true`.
     * @param  tensorArrayName        Name to use for the tensor array. Overrides the name used for the temporary tensor
     *                                array resource. If not provided or if an empty string is provided, then the name of
     *                                the created tensor array op is used, which is guaranteed to be unique.
     * @param  inferShape             Boolean value indicating whether shape inference is enabled. If `true`, all
     *                                elements must have the same shape.
     * @param  elementShape           Expected shape of the elements in the tensor array, if known. If this shape is not
     *                                fully defined, then gathering zero-sized tensor array elements will cause an error.
     * @param  colocateWithFirstWrite Boolean value indicating whether to place the tensor array on the same device as
     *                                the tensor used on its first write call (write operations include `write`,
     *                                `unstack`, and `split`). If `false`, the tensor array will be placed on the device
     *                                determined by the op creation context available during its initialization.
     * @param  name                   Name for the created tensor array ops.
     * @return Created tensor array.
     * @see "tensorflow.python.ops.tensor_array_ops._GraphTensorArray#__init__"
     */
    fun create(
        size: Output,
        dataType: DataType<*>,
        dynamicSize: Boolean = false,
        clearAfterRead: Boolean = true,
        tensorArrayName: String = "",
        inferShape: Boolean = true,
        elementShape: Shape = Shape(),
        colocateWithFirstWriteCall: Boolean = true,
        name: String = "TensorArray"
    ): TensorArray {
      val (handle, flow) = if (colocateWithFirstWriteCall)
        tf.device {
          tf.colocateWith(mutableSetOf(), ignore_existing = true) {
            tf.tensorArrayV3(size, dataType, elementShape, dynamicSize,
                             clearAfterRead, inferShape, tensorArrayName, name)
          }
        }
      else
        tf.tensorArrayV3(size, dataType, elementShape, dynamicSize,
                         clearAfterRead, inferShape, tensorArrayName, name)
      return createFromHandle(handle, flow, dataType, inferShape, elementShape, colocateWithFirstWriteCall)
    }
    
    /** Creates a tensor array from an existing tensor array handle.
     *
     * @param  handle                 Tensor handle to the tensor array.
     * @param  flow                   Float scalar tensor for the tensor array, used to control gradient flow.
     * @param  dataType               Data type of the elements in the tensor array.
     * @param  inferShape             Boolean value indicating whether shape inference is enabled. If `true`, all
     *                                elements must have the same shape.
     * @param  elementShape           Expected shape of the elements in the tensor array, if known. If this shape is not
     *                                fully defined, then gathering zero-sized tensor array elements will cause an error.
     * @param  colocateWithFirstWrite Boolean value indicating whether to place the tensor array on the same device as
     *                                the tensor used on its first write call (write operations include `write`,
     *                                `unstack`, and `split`). If `false`, the tensor array will be placed on the device
     *                                determined by the op creation context available during its initialization.
     * @return Created tensor array.
     */
    fun createFromHandle(
        handle: Output,
        flow: Output,
        dataType: DataType<*>,
        inferShape: Boolean = true,
        elementShape: Shape = Shape(),
        colocateWithFirstWriteCall: Boolean = true
    ): TensorArray =
        TensorArray(handle, flow, dataType, inferShape,
                    if (elementShape.isUnknown) null else elementShape,
                    colocateWithFirstWriteCall)
  }
}