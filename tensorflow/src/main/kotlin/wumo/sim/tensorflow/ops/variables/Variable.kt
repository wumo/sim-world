package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.base_dtype
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.core.Graph.Graph
import wumo.sim.tensorflow.core.ShapeMismatchException
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.gen.identity
import wumo.sim.tensorflow.ops.gen.variableV2
import wumo.sim.tensorflow.scope.NameScope.Companion.nameFromScopeName
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT32
import wumo.sim.tensorflow.types.types
import wumo.sim.util.Shape
import wumo.sim.util.a

class Variable(
    override val dataType: DataType<*>,
    private val variable: Output,
    private val initializeOp: Op,
    private val cachedValue: Output) : VariableLike {
  
  /** Graph where this variable is defined. */
  override val graph = variable.graph
  /** Name of this variable. */
  override val name
    get() = variable.op!!.name
  val device = variable.device
  override val shape = variable.shape
  val op = variable.op!!
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
  
  interface VariableGetter {
    operator fun invoke(
        name: String,
        dataType: DataType<*> = types.FLOAT32,
        shape: Shape? = null,
        initializer: Initializer? = null,
        regularizer: Regularizer? = null,
        trainable: Boolean = true,
        reuse: Reuse = ReuseOrCreateNew,
        collections: Set<Graph.Key<Variable>> = emptySet(),
        cachingDevice: DeviceFunction? = null,
        underlyingGetter: VariableGetter? = null
    ): Variable
  }
  
  companion object {
    /** Gets an existing variable with the specified name or creates a new one.
     *
     * This function prefixes the name with the current variable scope and performs variable reuse checks.
     *
     * TODO: Add example.
     *
     * @param  name          Variable name.
     * @param  dataType      Data type for the value of the created variable. If not provided, its value is inferred from
     *                       the provided initial value. If it cannot be inferred, then it will default to `FLOAT32`.
     * @param  shape         Shape for the value of the created variable. If `null`, an attempt will be made to infer the
     *                       shape of the variable from the provided initializer.
     * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
     *                       passed in the constructor is used. If that one is `null` too, then we use a new
     *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
     *                       variable separately.
     * @param  regularizer   Variable regularizer.
     * @param  trainable     If `true`, the default, the variable is added to the graph collection
     *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables
     *                       to use by the optimizers.
     * @param  reuse         [[Reuse]] value indicating whether to re-use an existing variable with the same name, create
     *                       a new variable, or do either.
     * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
     *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
     * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
     *                       to the variable's device. Typical use is to cache on the device where the ops using the
     *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
     * @return Requested variable.
     */
    internal fun getVariable(
        name: String,
        shape: Shape? = null,
        dataType: DataType<*>? = null,
        initializer: Initializer? = null,
        regularizer: Regularizer? = null,
        trainable: Boolean = true,
        reuse: Reuse = ReuseOrCreateNew,
        collections: Set<Graph.Key<Variable>> = emptySet(),
        cachingDevice: DeviceFunction? = null
    ): Variable =
        VariableScope.current.getVariable(
            VariableStore.current, name, shape, dataType, initializer, regularizer, trainable, reuse, collections,
            cachingDevice)
    
    /**
     * @see "tensorflow.python.ops.variables.Variable#__init__"
     */
    internal operator fun invoke(
        initializer: Initializer,
        shape: Shape? = null,
        dataType: DataType<*>? = null,
        trainable: Boolean = true,
        collections: Set<Graph.Key<Variable>> = emptySet(),
        cachingDevice: DeviceFunction? = null,
        name: String = "Variable"
    ): Variable {
      ops.init_scope {
        ops.name_scope(name) {
          val inferredDataType = dataType ?: initializer.dtype ?: FLOAT32
          val inferredShape = shape ?: initializer.shape ?: throw ShapeMismatchException(
              "No shape was provided for the new variable and it could not be inferred from the provided initializer.")
          val scopeName = ops.currentNameScope.scopeName
          val trueName = nameFromScopeName(scopeName)
//          //Use attr_scope and device(None) to simulate the behavior of
//          //colocate_with when the _variable we want to colocate with doesn't
//          //yet exist.
//          val attr = tensorflow.AttrValue()
//          attr.mutable_list().apply {
//            add_s("loc:@$trueName")
//          }
          val variableHandle = tf._variableV2(inferredShape, inferredDataType.cValue.base_dtype, name = scopeName)
          val initialValue = ops.name_scope("Initializer") {
            ops.colocate_with(variableHandle.op!!) {
              initializer(inferredShape, inferredDataType)
            }
          }
          val initializeOp = tf.assign(variableHandle,
                                       tryGuardAgainstUninitializedDependencies(name, initialValue))
          
        }
      }
      TODO()
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
     *
     * @see "tensorflow.python.ops.variables.Variable#_try_guard_against_uninitialized_dependencies"
     */
    private fun tryGuardAgainstUninitializedDependencies(variableName: String, initial_value: Output): Output {
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
      
      //Don't modify initial_value if it contains any cyclic dependencies.
      return if (has_cycle(initial_value.op!!, path = mutableSetOf()))
        initial_value
      else safe_initial_value_from_tensor(variableName, initial_value, mutableMapOf())
    }
    
    /**
     * Replace dependencies on variables with their initialized values.
     * @param initialValue A [Output]. The tensor to replace.
     * @param op_cache A dict mapping findOp names to [Op]s. Used to memoize
     * the results so as to avoid creating redundant operations.
     * @return A [Output] compatible with [initialValue]. Any inputs that lead to variable
     * values will be replaced with a corresponding graph that uses the
     * variable's initialized values. This is done on a best-effort basis. If no
     * modifications need to be made then [initialValue] will be returned unchanged.
     */
    private fun safe_initial_value_from_tensor(variableName: String, initialValue: Output, op_cache: MutableMap<String, Op>): Output {
      val op = initialValue.op
      val new_op = op_cache.compute(op!!.name) { _, new_op ->
        new_op ?: safe_initial_value_from_op(variableName, op, op_cache)
      }!!
      return new_op.outputs[initialValue.value_index]
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
    private fun safe_initial_value_from_op(variableName: String, op: Op, op_cache: MutableMap<String, Op>): Op {
      val op_type = op.opType
      if (op_type in a("IsVariableInitialized", "VarIsInitializedOp", "ReadVariableOp"))
        return op
      if (op_type in a("Variable", "VariableV2", "VarHandleOp")) {
        //Attempt to find the initialized_value of any variable reference / handles.
        val initialized_value = find_initialized_value_for_variable(op)
        return initialized_value?.op ?: op
      }
      //Recursively build initializer expressions for inputs.
      var modified = false
      val new_op_inputs = mutableListOf<Output>()
      for (op_input in op.inputs) {
        val new_op_input = safe_initial_value_from_tensor(variableName, op_input, op_cache)
        new_op_inputs += new_op_input
        modified = modified || (new_op_input != op_input)
      }
      //If at least one input was modified, replace the op.
      if (modified) {
        var new_op_type = op_type
        if (new_op_type == "RefSwitch")
          new_op_type = "Switch"
        val new_op_name = "${op.name}_$variableName".replace(":", "_")
        return buildOp(new_op_type, new_op_name) {
          new_op_inputs.forEach { addInput(it) }
          op.toNodeDef().attr().forEach { key, attrValue -> attr(key, attrValue) }
        }
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
      val variables = variable_op.graph.getCollection(Graph.Keys.GLOBAL_VARIABLES)
          .asSequence() + variable_op.graph.getCollection(Graph.Keys.LOCAL_VARIABLES)
      val var_names = variable_op.name
      val output_name = variable_op.outputs[0].name
      for (v in variables)
        if (v.name == var_names || v.name == output_name)
          return v.initialized_value()
      return null
    }
  }
}