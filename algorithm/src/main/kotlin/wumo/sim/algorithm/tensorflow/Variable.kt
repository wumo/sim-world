package wumo.sim.algorithm.tensorflow

import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.ops.control_flow_ops.cond
import wumo.sim.algorithm.tensorflow.ops.gen.identity
import wumo.sim.algorithm.tensorflow.ops.variables.VariableLike
import wumo.sim.algorithm.tensorflow.types.DataType
import wumo.sim.util.a

class Variable(
    override val dataType: DataType<*>,
    private val variable: Output,
    private val initializeOp: Op,
    private val cachedValue: Output,
    op: Op, value_index: Int) : VariableLike {
  /** Graph where this variable is defined. */
  override val graph = variable.graph
  /** Name of this variable. */
  override val name
    get() = variable.op!!.name
  val device = variable.device
  override val shape = variable.shape
  override val value
    get() = snapshot
  override val initializer: Op
    get() = TODO("not implemented")
  override val isInitialized: Output
    get() = TODO("not implemented")
  
  override val initializedValue: Output
    get() =
      ops.init_scope {
        tf.cond(is_variable_initialized(), { read_value() }, { initial_value })
      }
  
  override fun read(name: String): Output {
    TODO("not implemented")
  }
  
  override fun gather(indices: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assign(value: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assignAdd(value: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assignSub(value: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assignScatter(indices: Output, values: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assignScatterAdd(indices: Output, values: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assignScatterSub(indices: Output, values: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun toTensor(): Output {
    TODO("not implemented")
  }
  
  lateinit var initial_value: Output
  lateinit var initializer_op: Output
  lateinit var snapshot: Output
  /**
   * Returns the value of the initialized variable.
   *
   * You should use this instead of the variable itself to initialize another
   * variable with a value that depends on the value of this variable.
   *
   * @return A `Output` holding the value of this variable after its initializer has run.
   */
  fun initialized_value() =
      tf.init_scope {
        tf.cond(is_variable_initialized(), { read_value() }, { initial_value })
      }
  
  /**
   * Tests if a variable has been initialized.
   *
   * @return Returns a scalar boolean Output, `True` if the variable has been
   * initialized, `False` otherwise.
   */
  private fun is_variable_initialized() =
      tf.is_variable_initialized(this)
  
  fun read_value() = tf.identity(this, name = "read")
  fun assign(value: Output, use_locking: Boolean = false): Output {
    return tf.assign(this, value)
  }
  
  /**
   * Attempt to guard against dependencies on uninitialized variables.
   *
   * Replace references to variables in `initial_value` with references to the
   * variable's initialized values. The initialized values are essentially
   * conditional TensorFlow graphs that return a variable's value if it is
   * initialized or its `initial_value` if it hasn't been initialized. This
   * replacement is done on a best effort basis:
   *
   * - If the [initial_value] graph contains cycles, we don't do any
   * replacements for that graph.
   * - If the variables that [initial_value] depends on are not present in the
   * `GLOBAL_VARIABLES` or `LOCAL_VARIABLES` we don't replace them.
   *
   * In these cases, it is up to the caller to ensure that the `initial_value`
   * graph uses initialized variables or that they guard access to variables
   * using their `initialized_value` method.
   * @param initial_value The initial value.
   * @return A [Output] suitable to initialize a variable.
   */
  fun try_guard_against_uninitialized_dependencies(initial_value: Output): Output {
    /**Detect cycles in the dependencies of [initial_value].*/
    fun has_cycle(op: Op, path: MutableSet<String>): Boolean {
      if (op.name in path) return true
      path += op.name
      for (op_input in op.inputs)
        if (has_cycle(op_input.op!!, path))
          return true
      for (op_control_input in op.controlInputs)
        if (has_cycle(op_control_input, path))
          return true
      path.remove(op.name)
      return false
    }
    if (has_cycle(initial_value.op!!, path = mutableSetOf()))
      return initial_value
    return safe_initial_value_from_tensor(initial_value, mutableMapOf())
  }
  
  /**
   * Replace dependencies on variables with their initialized values.
   * @param tensor A [Output]. The tensor to replace.
   * @param op_cache A dict mapping findOp names to [Op]s. Used to memoize
   * the results so as to avoid creating redundant operations.
   * @return A [Output] compatible with [tensor]. Any inputs that lead to variable
   * values will be replaced with a corresponding graph that uses the
   * variable's initialized values. This is done on a best-effort basis. If no
   * modifications need to be made then [tensor] will be returned unchanged.
   */
  private fun safe_initial_value_from_tensor(tensor: Output, op_cache: MutableMap<String, Op>): Output {
    val op = tensor.op
    val new_op = op_cache.compute(op!!.name) { _, new_op ->
      new_op ?: safe_initial_value_from_op(op, op_cache)
    }!!
    return new_op.outputs[tensor.value_index]
  }
  
  /**
   * Replace dependencies on variables with their initialized values.
   * @param op A [Op]. The tensor to replace.
   * @param op_cache A dict mapping findOp names to [Op]s. Used to memoize
   * the results so as to avoid creating redundant operations.
   * @return A [Output] compatible with [op]. Any inputs that lead to variable
   * values will be replaced with a corresponding graph that uses the
   * variable's initialized values. This is done on a best-effort basis. If no
   * modifications need to be made then [op] will be returned unchanged.
   */
  private fun safe_initial_value_from_op(op: Op, op_cache: MutableMap<String, Op>): Op {
    val op_type = op.opType
    if (op_type in a("IsVariableInitialized", "VarIsInitializedOp", "ReadVariableOp"))
      return op
    //Attempt to find the initialized_value of any variable reference / handles.
    if (op_type in a("Variable", "VariableV2", "VarHandleOp")) {
      val initialized_value = find_initialized_value_for_variable(op)
      return initialized_value?.op ?: op
    }
    //Recursively build initializer expressions for inputs.
    var modified = false
    val new_op_inputs = mutableListOf<Output>()
    for (op_input in op.inputs) {
      val new_op_input = safe_initial_value_from_tensor(op_input, op_cache)
      new_op_inputs += new_op_input
      modified = modified || (new_op_input != op_input)
    }
    //If at least one input was modified, replace the op.
    if (modified) {
      var new_op_type = op_type
      if (new_op_type == "RefSwitch")
        new_op_type = "Switch"
      var new_op_name = op.name + "_" + this.op!!.name
      new_op_name = new_op_name.replace(":", "_")
      return tf.g.create_op(new_op_type, new_op_inputs, op.output_types, name = new_op_name, attrs = op.attr)
    }
    return op
  }
  
  /**
   * Find the initialized value for a variable op.
   *
   * To do so, lookup the variable op in the variables collection.
   * @param variable_op
   * @return A [Output] representing the initialized value for the variable or `null`
   * if the initialized value could not be found.
   */
  private fun find_initialized_value_for_variable(variable_op: Op): Output? {
    val var_names = variable_op.name
    for (v in tf.global_variables)
      if (v.op!!.name == var_names)
        return v.initialized_value()
    return null
  }
}