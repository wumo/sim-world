package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.core.Graph.Graph
import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.tensorflow.core.ShapeMismatchException
import wumo.sim.tensorflow.createOp
import wumo.sim.tensorflow.ops.DeviceFunction
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.ops
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.types
import wumo.sim.util.Shape

class Variable(
    override val dataType: DataType<*>,
    val variable: Output,
    initializeOp: Op,
    private val initialValue: Output,
    snapshot: Output) : VariableLike {
  
  override val graph = variable.graph
  override val name = variable.op.name
  val device = variable.device
  override val shape = variable.shape
  val op = variable.op
  override val value = snapshot
  fun readValue() = tf.identity(variable, name = "read")
  override val initializer = initializeOp
  override val isInitialized: Output
    get() = tf.with(graph) {
      tf.isVariableInitialized(variable)
    }
  
  override val initializedValue: Output
    get() =
      tf.init_scope {
        tf.cond(isInitialized, { readValue() }, { initialValue })
      }
  
  /** Contains the partition/save-slice information for this variable. */
  internal var partitionInformation: PartitionInformation? = null
  
  override fun read(name: String): Output =
      tf.identity(variable, name = "read")
  
  override fun gather(indices: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assign(value: Output, name: String): Output =
      tf.assign(variable, value)
  
  override fun assignAdd(value: Output, name: String): Output =
      tf.assignAdd(variable, value, name = name)
  
  override fun assignSub(value: Output, name: String): Output =
      tf.assignSub(variable, value, name = name)
  
  override fun assignScatterSub(indices: Output, values: Output, use_locking: Boolean, name: String): Output {
    if (values.dataType != dataType)
      throw InvalidDataTypeException("Expected '$dataType', but got '${values.dataType}'.")
    return tf.scatterSub(variable, indices, values, use_locking, name)
  }
  
  override fun toString() = op.toString()
  override fun equals(other: Any?) =
      when (other) {
        is Variable -> op == other.op
        else -> false
      }
  
  override fun hashCode() = op.hashCode()
  
  interface VariableGetter {
    operator fun invoke(
        name: String,
        dataType: DataType<*>? = types.FLOAT,
        shape: Shape? = null,
        initializer: Initializer? = null,
        regularizer: Regularizer? = null,
        trainable: Boolean = true,
        reuse: Reuse = ReuseOrCreateNew,
        collections: MutableSet<Graph.Key<Variable>> = mutableSetOf(),
        cachingDevice: DeviceFunction? = null,
        underlyingGetter: VariableGetter? = null
    ): Variable
  }
  
  /** Class that contains partitioning information for a variable that can also be used to save it as a slice.
   *
   * @param  fullName         Name of the full variable, of which the variable is a partition.
   * @param  fullShape        Shape of the full variable, of which the variable is a partition.
   * @param  partitionOffsets Offsets of the partition into the full variable.
   * @param  partitionShape   Shape of the variable.
   */
  class PartitionInformation(
      val fullName: String,
      val fullShape: Shape,
      val partitionOffsets: IntArray,
      val partitionShape: IntArray
  )
  
  companion object {
    /** Gets an existing variable with the specified name or creates a new one.
     *
     * This function prefixes the name with the current variable scope and performs variable reuse checks.
     *
     * TODO: Add example.
     *
     * @param  name          Variable name.
     * @param  dataType      Data type for the value of the created variable. If not provided, its value is inferred from
     *                       the provided initial value. If it cannot be inferred, then it will default to `FLOAT`.
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
        collections: MutableSet<Graph.Key<Variable>> = mutableSetOf(),
        cachingDevice: DeviceFunction? = null
    ): Variable =
        VariableScope.current.getVariable(
            VariableStore.current, name, shape, dataType, initializer, regularizer, trainable, reuse, collections,
            cachingDevice)
    
    val globalVariables: Set<Variable> = tf.currentGraph.globalVariables
    
    val localVariables: Set<Variable> = tf.currentGraph.localVariables
    
    /** Creates an op that initializes the provided variables.
     *
     * After you launch the graph in a session, you can run the returned op to initialize all the variables in
     * `variables`. This op runs all the initializers of the variables in `variables`, in parallel.
     *
     * Calling [initializer] is equivalent to passing the list of initializers to [tf.group].
     *
     * If [variables] is empty, the method still returns an op that can be run. That op has no effect (i.e., it is a
     * [tf.noOp]).
     *
     * @param  variables Set of variables to initialize.
     * @param  name      Name for the created op.
     * @return Created op.
     */
    fun initializer(variables: Set<Variable>, name: String = "init"): Op =
        if (variables.isNotEmpty())
          tf.group(variables.mapTo(mutableSetOf()) { it.initializer }, name)
        else
          tf.noOp(name)
    
    /**
     * @see "tensorflow.python.ops.variables.Variable#__init__"
     */
    internal operator fun invoke(
        initializer: Initializer,
        shape: Shape? = null,
        dataType: DataType<*>? = null,
        trainable: Boolean = true,
        collections: MutableSet<Graph.Key<Variable>>? = null,
        cachingDevice: DeviceFunction? = null,
        name: String = "Variable"
    ): Variable =
        tf.init_scope {
          tf.nameScope(name) {
            val inferredDataType = dataType ?: initializer.dataType ?: FLOAT
            val inferredShape = shape ?: initializer.shape ?: throw ShapeMismatchException(
                "No shape was provided for the new variable and it could not be inferred from the provided initializer.")
            val scopeName = tf.currentNameScope
            val trueName = ops.convertNameScopeToName(scopeName)
//          //Use attrScope and device(None) to simulate the behavior of
//          //colocateWith when the _variable we want to colocate with doesn't
//          //yet exist.
//          val attr = tensorflow.AttrValue()
//          attr.mutable_list().apply {
//            add_s("loc:@$trueName")
//          }
            val variableHandle = tf.variableV2(inferredShape, inferredDataType.baseDataType,
                                                sharedName = trueName, name = scopeName)
            val initialValue = tf.nameScope("Initializer") {
              tf.colocateWith(variableHandle.op) {
                initializer(inferredShape, inferredDataType)
              }
            }
            val initializeOp = tf.assign(variableHandle,
                                          tryGuardAgainstUninitializedDependencies(variableHandle.name, initialValue))
            val snapshot = if (cachingDevice != null)
              tf.device(cachingDevice) {
                tf.identity(variableHandle, name = "read")
              }
            else
              tf.colocateWith(variableHandle.op) {
                tf.identity(variableHandle, name = "read")
              }
            val createdVariable = Variable(inferredDataType, variableHandle, initializeOp.op, initialValue, snapshot)
            val _collections = collections ?: mutableSetOf()
            if (_collections.isEmpty())
              _collections.add(Graph.Keys.GLOBAL_VARIABLES)
            if (trainable)
              _collections.add(Graph.Keys.TRAINABLE_VARIABLES)
            _collections.forEach { key -> createdVariable.graph.addToCollection(createdVariable, key) }
            createdVariable
          }
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
          if (has_cycle(op_input.op, path))
            return true
        for (op_control_input in op.controlInputs)
          if (has_cycle(op_control_input, path))
            return true
        path.remove(op.name)
        return false
      }
      
      //Don't modify initial_value if it contains any cyclic dependencies.
      return if (has_cycle(initial_value.op, path = mutableSetOf()))
        initial_value
      else safeInitialValueFromTensor(variableName, initial_value, mutableMapOf())
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
    private fun safeInitialValueFromTensor(variableName: String, initialValue: Output, op_cache: MutableMap<String, Op>): Output {
      val op = initialValue.op
      val new_op = op_cache.compute(op.name) { _, new_op ->
        new_op ?: safeInitialValueFromOp(variableName, op, op_cache)
      }!!
      return new_op.outputs[initialValue.valueIndex]
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
    private fun safeInitialValueFromOp(variableName: String, op: Op, op_cache: MutableMap<String, Op>): Op {
      val op_type = op.opType
      return when (op_type) {
        "IsVariableInitialized", "VarIsInitializedOp", "ReadVariableOp" ->
          op
        "Variable", "VariableV2", "VarHandleOp" -> {
          //Attempt to find the initialized_value of any variable reference / handles.
          val initialized_value = findInitializedValueForVariable(op)
          initialized_value?.op ?: op
        }
        else -> {
          //Recursively build initializer expressions for inputs.
          var modified = false
          val new_op_inputs = op.inputs.map { op_input ->
            val new_op_input = safeInitialValueFromTensor(variableName, op_input, op_cache)
            modified = modified || (new_op_input != op_input)
            new_op_input
          }
          //If at least one input was modified, replace the op.
          if (modified) {
            var new_op_type = op_type
            if (new_op_type == "RefSwitch")
              new_op_type = "Switch"
            val new_op_name = "${op.name}_$variableName".replace(":", "_")
            createOp(new_op_type, new_op_name, new_op_inputs, op.nodeDef().attrMap)
//            buildOp(new_op_type, new_op_name) {
//              val nodeDef = op.toNodeDef()
//              createOp(new_op_name, new_op_inputs, nodeDef)
//              new_op_inputs.forEach { addInput(it) }
//              op.toNodeDef().attr().forEach { key, attrValue -> attr(key, attrValue) }
//            }
          } else
            op
        }
      }
    }
    
    /**
     * Find the initialized value for a variable op.
     *
     * To do so, lookup the variable op in the variables collection.
     * @param variable_op
     * @return A [Output] representing the initialized value for the variable or `null`
     * if the initialized value could not be found.
     */
    private fun findInitializedValueForVariable(variable_op: Op): Output? {
      val variables = variable_op.graph.getCollection(Graph.Keys.GLOBAL_VARIABLES)
          .asSequence() + variable_op.graph.getCollection(Graph.Keys.LOCAL_VARIABLES)
      val var_names = variable_op.name
      val output_name = variable_op.outputs[0].name
      for (v in variables)
        if (v.name == var_names || v.name == output_name)
          return v.initializedValue
      return null
    }
  }
}