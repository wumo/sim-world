package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.ops.HasName
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputConvertible
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

interface VariableLike : OutputConvertible,HasName {
  /** Graph where this variable is val ined. */
  val graph: Graph
  
  /** Data type of this variable. */
  val dataType: DataType<*>
  
  /** Shape of this variable. */
  val shape: Shape
  
  /** Returns a cached op which reads the last value of this partitioned variable.
   *
   * You can not assign a new value to the returned tensor as it is not a reference to the variable.
   *
   * The returned op output will not inherit the control dependencies from the native where the value is used, which is
   * equivalent behavior to that of getting the value of a variable.
   *
   * NOTE: You usually do not need to call this method directly, as all ops that use variables do so by internally
   * converting them to tensors.
   */
  val value: Output
  
  /** Op responsible for initializing this variable. */
  val initializer: Op
  
  /** Op output that is `true` when the variable has been initialized and `false` otherwise. */
  val isInitialized: Output
  
  /** Value of the initialized variable. You should use this instead of the variable itself to initialize
   * another variable with a value that depends on the value of this variable.
   *
   * Example:
   * ```
   *   // Initialize `v` with random values, and then use `initializedValue` to guarantee that `v` has been initialized
   *   // before its value is used to initialize `w`. The random tensor will only be sampled once.
   *   val v = tf.variable("v", FLOAT, Shape(10, 40), tf.RandomTruncatedNormalInitializer())
   *   val w = tf.variable("w", initializer = tf.ConstantInitializer(v.initializedValue * 2.0))
   * ```
   */
  val initializedValue: Output
  
  /** Creates an op that reads the value of this variable.
   *
   * This method should be used when there are multiple reads, or when it is desirable to read the value only after
   * some condition is true.
   *
   * The returned value may be different from that of [[value]] depending on the device being used, the control
   * dependencies, etc.
   *
   * @return Created op.
   */
  fun read(name: String = "Read"): Output
  
  /** Creates an op that reads the value of this variable sparsely, using the provided `indices`.
   *
   * This method should be used when there are multiple reads, or when it is desirable to read the value only after
   * some condition is true.
   *
   * @param  indices Indices to use for the sparse read.
   * @param  name    Name for the created op.
   * @return Created op.
   */
  fun gather(indices: Output, name: String = "Gather"): Output
  
  /** Creates an op that assigns the provided value to this variable and returns its value.
   *
   * @param  value Value to assign the variable to.
   * @param  name  Name for created op.
   * @return Variable value read op, after the assignment.
   */
  fun assign(value: Output, name: String = "Assign"): Output
  
  /** Creates an op that adds the provided value to the current value of the variable and returns its value.
   *
   * @param  value Value to add to the current variable value.
   * @param  name  Name for created op.
   * @return Variable value read op, after the addition.
   */
  fun assignAdd(value: Output, name: String = "AssignAdd"): Output
  
  /** Creates an op that subtracts the provided value from the current value of the variable and returns its value.
   *
   * @param  value Value to subtract from the current variable value.
   * @param  name  Name for created op.
   * @return Variable value read op, after the subtraction.
   */
  fun assignSub(value: Output, name: String = "AssignAdd"): Output
  
  /** Creates an op that subtracts the provided sparse value from the current value of the variable and returns its
   * value.
   *
   * @param  indices Indices corresponding to the `values` being subtracted.
   * @param  values  Values to be subtracted, corresponding to the provided `indices`.
   * @param  name    Name for created op.
   * @return Variable value read op, after the subtraction.
   */
  fun assignScatterSub(indices: Output, values: Output, use_locking: Boolean = false, name: String = "ScatterSub"): Output
  
  /** Converts this variable to an op output. This function simply returns an op corresponding to the variable value. */
  override fun toOutput() = value
}