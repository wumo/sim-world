package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.AttrValue
import org.bytedeco.javacpp.tensorflow.TF_Version
import org.tensorflow.framework.GraphDef
import wumo.sim.tensorflow.Session
import wumo.sim.tensorflow.core.*
import wumo.sim.tensorflow.ops.control_flow_ops.CondContext
import wumo.sim.tensorflow.ops.control_flow_ops.ControlFlowContext
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops
import wumo.sim.tensorflow.ops.gen.gen_ops
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.ops.variables.VariableScope
import wumo.sim.tensorflow.ops.variables.variables
import wumo.sim.util.DynamicVariable
import wumo.sim.util.lazyLogger
import wumo.sim.util.println

object ops {
  val logger by lazyLogger()
  
  const val COLOCATION_OPS_ATTRIBUTE_NAME = "_class"
  const val COLOCATION_OPS_ATTRIBUTE_PREFIX = "loc:@"
  val VALID_OP_NAME_REGEX = Regex("^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$")
  val VALID_NAME_SCOPE_REGEX = Regex("^[A-Za-z0-9_.\\-/]*$")
  
  data class GraphConstructionScope(
      val graph: Graph = Graph(),
      var nameScope: String = "",
      var device: String = "",
      var deviceFunction: (OpSpecification) -> String = { it.device },
      val colocationOps: MutableSet<Op> = mutableSetOf(),
      val controlDependencies: MutableSet<Op> = mutableSetOf(),
      val attributes: MutableMap<String, tensorflow.AttrValue> = mutableMapOf(),
      var container: String = "", // TODO: !!! Use containers.
      var controlFlowContext: ControlFlowContext? = null,
      var outerContext: GraphConstructionScope? = null)
  
  internal val graphConstructionScope = DynamicVariable(GraphConstructionScope(core.defaultGraph))
  
  /** Checks whether the provided string is a valid op name.
   *
   * @param  name String to check.
   * @return Boolean value indicating whether the check was successful.
   */
  internal fun checkName(name: String) = VALID_OP_NAME_REGEX.matches(name)
  
  /** Checks whether the provided string is a valid name scope for creating ops.
   *
   * @param  nameScope String to check.
   * @return Boolean value indicating whether the check was successful.
   */
  internal fun checkNameScope(nameScope: String) = VALID_NAME_SCOPE_REGEX.matches(nameScope)
  
  /** Converts the provided name scope to a valid op name, by removing a trailing `"/"` if there exists one.
   *
   * @param  nameScope Name scope to convert.
   * @return Name obtained from the provided name scope.
   */
  internal fun convertNameScopeToName(nameScope: String) =
      if (nameScope.endsWith('/'))
        nameScope.substring(0, nameScope.lastIndex)
      else
        nameScope
  
  interface API :
      gen_ops,
      array_ops.API,
      clip_ops.API,
      control_flow_ops.API,
      const_ops.API,
      data_flow_ops.API,
      gradient_ops.API,
      init_ops.API,
      math_ops.API,
      nn_ops.API,
      no_op.API,
      random_ops.API,
      state_ops.API,
      variables.API {
    
    /** Returns the graph of the current op creation context. */
    val currentGraph get() = graphConstructionScope.value.graph
    /** Returns the name scope of the current op creation context. */
    val currentNameScope get() = graphConstructionScope.value.nameScope
    /** Returns the device of the current op creation context. */
    val currentDevice get() = graphConstructionScope.value.device
    /** Returns the device function of the current op creation context. */
    val currentDeviceFunction get() = graphConstructionScope.value.deviceFunction
    /** Returns the colocation ops of the current op creation context. */
    val currentColocationOps get() = graphConstructionScope.value.colocationOps
    /** Returns the control dependencies of the current op creation context. */
    val currentControlDependencies get() = graphConstructionScope.value.controlDependencies
    /** Returns the attributes of the current op creation context. */
    val currentAttributes get() = graphConstructionScope.value.attributes
    /** Returns the container of the current op creation context. */
    val currentContainer get() = graphConstructionScope.value.container
    /** Returns the control flow context of the current op creation context. */
    val currentControlFlowContext get() = graphConstructionScope.value.controlFlowContext

//  val g = Graph()
//  val trainables = mutableListOf<Variable>()
//  val global_variables = mutableListOf<Variable>()
//  val train_ops = mutableListOf<Op>()
//  private val rootNs = NameScope("$", null)
//  private val rootVs = VariableScope("$", rootNs)
//  var ctxNs = rootNs
//  var ctxVs = rootVs

//  var device: String = ""
//  var colocate_with = ArrayDeque<Op>()
//  val control_ops = ArrayDeque<Op>()
//  val attr_scope_map = hashMapOf<String, tensorflow.AttrValue>()

//    lateinit var session: Session
    /**
    A context manager that lifts ops out of control-flow scopes and function-building graphs.
    
    There is often a need to lift variable initialization ops out of control-flow
    scopes, function-building graphs, and gradient tapes. Entering an
    [init_scope] is a mechanism for satisfying these desiderata. In particular,
    entering an [init_scope] has three effects:
    
    (1) All control dependencies are cleared the moment the scope is entered;
    this is equivalent to entering the context manager returned from
    `control_dependencies(None)`, which has the side-effect of exiting
    control-flow scopes like `tf.cond` and `tf.while_loop`.
    
    (2) All operations that are created while the scope is active are lifted
    into the lowest context on the `context_stack` that is not building a
    graph function. Here, a context is defined as either a graph or an eager
    context. Every context switch, i.e., every installation of a graph as
    the default graph and every switch into eager mode, is logged in a
    thread-local stack called `context_switches`; the log entry for a
    context switch is popped from the stack when the context is exited.
    Entering an `init_scope` is equivalent to crawling up
    `context_switches`, finding the first context that is not building a
    graph function, and entering it. A caveat is that if graph mode is
    enabled but the default graph stack is empty, then entering an
    `init_scope` will simply install a fresh graph as the default one.
    
    (3) The gradient tape is paused while the scope is active.
     */
    fun <R> init_scope(block: () -> R): R {
      try {
        return control_dependencies {
          block()
        }
      } finally {
      }
    }
    
    /** Creates a context that can be used for creating ops according to the provided options.
     *
     * = General Information =
     *
     * During graph creation, a context is maintained that includes:
     *   - The current graph in which new ops are placed.
     *   - The current name scope used for naming these new ops.
     *   - A device function, used to decide in which device (e.g., CPU) the new ops should be placed and executed.
     *   - A set of colocation ops for the newly constructed ops. This means that the newly created ops will be placed on
     * the same device as these colocation ops.
     *   - A set of ops defining control dependencies for the newly constructed ops. This means that the newly
     * constructed ops are constrained to only execute after the provided set of ops has finished executing.
     *   - A map from op attribute names to values for the newly constructed ops. These attributes will be applied to all
     * newly constructed ops.
     *   - A container name for the newly constructed resource ops. All newly constructed resource ops will be placed in
     * the provided container.
     *
     * Note that all arguments of this function are optional. If they are not provided, then the corresponding option in
     * current op creation context is left unchanged.
     *
     * Care must be taken if concurrency is used while creating the graph because the op creation context is wrapped
     * inside a [DynamicVariable]. More information on this general issue can be found at
     * [[http://stevenskelton.ca/threadlocal-variables-scala-futures/]].
     *
     * = Argument Specifics =
     *
     * == Graph ==
     *
     * When `with(...)` is used with a graph, then all ops created within its code block will be placed in the
     * provided graph.
     *
     * For example:
     * ```
     *   val g = Graph()
     *   createWith(graph = g) {
     *     val c = constant(5.0)
     *     assert(c.graph == g)
     *   }
     * ```
     *
     * == Name Scope ==
     *
     * When `createWith(...)` is used with a name scope, the provided name scope is appended to the context name scope,
     * generating a new op creation context. This new context is used for all ops created within the code block provided
     * in the `createWith(...)` function. The `nameScope` argument will be interpreted as follows:
     *
     *   - A string not ending with `"/"` will create a new name scope, in which `nameScope` is appended to the prefix of
     *     all operations created in the provided code block. If `nameScope` has been used before, it will be made unique
     *     by calling `uniqueName(graph = context.graph, name = nameScope)`.
     *   - A string ending with `"/"` will be treated as an "absolute" name scope, which makes it possible to re-enter
     *     existing scopes. Such absolute name scopes can be obtained by using the `currentNameScope` function, from
     *     within the appropriate context.
     *   - A value of `""` will reset the current name scope to the top-level (i.e., empty) name scope.
     *
     * This function checks the provided `nameScope` for validity by checking whether it matches: (i) the regular
     * expression `[A-Za-z0-9.][A-Za-z0-9_.\\-/]*` if the current context name scope is empty (i.e., at the root), or
     * (ii) the regular expression `[A-Za-z0-9_.\\-/]*`, otherwise.
     *
     * For example:
     * ```
     *   // No name scope used
     *   val c = constant(1.0, name = "C")
     *   assert(c.op.name == "C")
     *   val c1 = constant(2.0, name = "C_1")
     *   assert(c_1.op.name == "C_1")
     *
     *   // Create a name scope called "Nested"
     *   createWith(nameScope = "Nested") {
     *     val nameScope = currentNameScope
     *     val nestedC = constant(3.0, name = "C")
     *     assert(nestedC.op.name == "Nested/C")
     *
     *     // Create a nested name scope called "Inner"
     *     createWith(nameScope = "Inner") {
     *       val nestedInnerC = constant(4.0, name = "C")
     *       assert(nestedInnerC.op.name == "Nested/Inner/C")
     *     }
     *
     *     // Create a nested name scope called "Inner_1"
     *     createWith(nameScope = "Inner_1") {
     *       val nestedInner1C = constant(5.0, name = "C")
     *       assert(nestedInner1C.op.name == "Nested/Inner_1/C")
     *
     *       createWith(nameScope = nameScope) {
     *         val nestedC1 = constant(6.0, name = "C_1")
     *         assert(nestedC1.op.name == "Nested/C_1")
     *
     *         // Reset the name scope using ""
     *         createWith(nameScope = "") {
     *           val c2 = constant(7.0, name = "C_2")
     *           assert(c2.op.name == "C_2")
     *         }
     *       }
     *     }
     *   }
     * ```
     *
     * == Device ==
     *
     * When `createWith(...)` is used with a device, a `deviceFunction` argument can be additionally used (aside from the
     * device string representation provided through the `device` argument), that is a function taking an
     * [OpSpecification] as input and returning a string representation of the device where the corresponding op should
     * be placed. This function is invoked every time a new op is created within the provided code block. If the function
     * returns `null` for some op, then all subsequent invocations of `createWith(deviceFunction = ...)` in the provided
     * code block will be ignored. For information about the valid syntax of device name strings, see the documentation
     * in [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).
     *
     * Note that the device scope may be overridden by op wrappers or other library code. For example, a variable
     * assignment op must be colocated with the corresponding variable. Incompatible device scopes will be ignored.
     *
     * For example:
     * ```
     *   // Specifying which device to use
     *   createWith(device = "/GPU:0") {
     *     // All ops constructed in this code block will be placed in GPU 0
     *     val gpu0C = constant(7.0)
     *     assert(gpu0C.device == "/device:GPU:0")
     *
     *     // Reset the device being used
     *     createWith(device = null) {
     *       // All ops constructed in this code block will have no assigned device
     *       val c = constant(8.0)
     *       assert(c.device == "")
     *     }
     *   }
     *
     *   // Using a device function
     *   def matmulOnGPU(opSpecification: OpSpecification): String = {
     *     if (opSpecification.opType == "MatMul")
     *       "/GPU:0"
     *     else
     *       "/CPU:0"
     *   }
     *
     *   createWith(deviceFunction = matmulOnGPU) {
     *     // All ops of type "MatMul" constructed in this code block will be placed on GPU 0. All other operations will
     *     // be placed on CPU 0.
     *     val c = constant(9.0)
     *     assert(c.device == "/device:CPU:0")
     *     val m = matmul(c, constant(10.0))
     *     assert(m.device == "/device:GPU:0")
     *   }
     * ```
     *
     * == Colocation Ops ==
     *
     * When `createWith(...)` is used with a set of colocation ops, then all ops created within its code block will be
     * placed on the same device as the provided colocation ops. Note that if a set of colocation ops already exists in
     * the current op creation context (e.g., as the result of nesting multiple `createWith(colocationOps = ...)` calls),
     * then the new set of colocation ops will be the union of the two sets. If provided an empty colocation ops set,
     * then the new set of colocation ops will also be empty (i.e., it is being reset).
     *
     * Note that using a non-empty set of colocation ops resets any existing device constraints. In other words,
     * colocation ops override any other device placement specification.
     *
     * For example:
     * ```
     *   val a = createWith(device = "/CPU:0")(constant(1.0))
     *   val b = createWith(device = "/GPU:0")(constant(1.0))
     *   assert(a.colocationOps === Set.empty[Op])
     *   assert(b.colocationOps === Set.empty[Op])
     *   val c = createWith(colocationOps = Set(a))(constant(1.0))
     *   assert(c.colocationOps === Set[Op](a))
     *   createWith(colocationOps = Set[Op](b)) {
     *     val d = constant(1.0)
     *     assert(d.colocationOps === Set[Op](b))
     *     createWith(colocationOps = Set[Op](a, d)) {
     *       val e = constant(1.0)
     *       assert(e.colocationOps === Set[Op](a, b, d))
     *       createWith(colocationOps = Set.empty[Op]) {
     *         val f = constant(1.0)
     *         assert(f.colocationOps === Set.empty[Op])
     *       }
     *     }
     *   }
     * ```
     *
     * == Control Dependencies ==
     *
     * When `createWith(...)` is used with a set of control dependencies, then all ops created within its code block will
     * be dependent on the control dependency ops. This means that they will be guaranteed to execute only after all of
     * the control dependencies ops have finished executing. Note that if a set of control dependencies already exists in
     * the current op creation context (e.g., as the result of nesting multiple `createWith(controlDependencies = ...)`
     * calls), then the new set of control dependencies will be the union of the two sets. Furthermore, if an empty set
     * is provided, then the control dependencies are cleared, instead of taking the union with the current control
     * dependencies.
     *
     * For example:
     * ```
     *   val a = constant(1.0)
     *   val b = constant(1.0)
     *   createWith(controlDependencies = Set(a)) {
     *     val c = constant(1.0)
     *     assert(c.controlInputs.toSet == Set(a))
     *     createWith(controlDependencies = Set(b, c)) {
     *       val d = constant(1.0)
     *       assert(d.controlInputs.toSet == Set(a, b, c))
     *       createWith(controlDependencies = Set()) {
     *         createWith(controlDependencies = Set(d)) {
     *           val e = constant(1.0)
     *           assert(e.controlInputs.toSet == Set(d))
     *         }
     *       }
     *     }
     *   }
     *   assert(a.controlOutputs.toSet == Set(c, d))
     *   assert(b.controlOutputs.toSet == Set(d))
     *   assert(c.controlOutputs.toSet == Set())
     *   assert(d.controlOutputs.toSet == Set(e))
     *   assert(e.controlOutputs.toSet == Set())
     * ```
     *
     * Note that transitive dependencies are eliminated (e.g., if `a` depends on `b` and `c`, and `b` depends on `c`,
     * then the dependency of `a` on `c` is ignored) in order not to add redundant control dependencies to the graph.
     *
     * == Attributes ==
     *
     * When `createWith(...)` is used with a set of attributes, then all ops created within its code block will have
     * those attributes set to the provided values when constructed. Note that if a map from attribute names to values
     * already exists in the current op creation context, then the two maps are merged. If a name exists in both, then
     * the provided value overrides the existing one, otherwise, the union of the two maps is used. Note that if the
     * value for an attribute in the provided map is set to `null`, then that attribute name-value pair is completely
     * removed from the op creation context.
     *
     * For example:
     * ```
     *   val a = constant(1.0)
     *   assert(a.stringAttribute("_a") == null)
     *   createWith(attributes = Map("_a" -> "foo")) {
     *     val b = constant(1.0)
     *     assert(b.stringAttribute("_a") == "foo")
     *     createWith(attributes = Map("_a" -> "bar")) {
     *       val c = constant(1.0)
     *       assert(c.stringAttribute("_a") == "bar")
     *       createWith(attributes = Map("_a" -> null)) {
     *         val d = constant(1.0)
     *         assert(d.stringAttribute("_a") == null)
     *       }
     *     }
     *   }
     * ```
     *
     * == Container ==
     *
     * Stateful operations, such as variables and queues, can maintain their states on devices so that they can be shared
     * by multiple processes. A resource container is a string name under which these stateful operations are tracked.
     * These resources can be released or cleared with `Session.reset`. // TODO: [SESSION] Add that function reference.
     *
     * When `createWith(...)` is used with a container, then all resource ops created within its code block will be
     * placed in the provided container. A new value for the container always overrides the previous value, except if
     * `null`, meaning that the previous value is used. The default root container name is `""`.
     *
     * TODO: [VARIABLE] Add example when we have support for variables.
     *
     * == Combining Arguments ==
     *
     * Multiple arguments can be provided to change several aspects of the current op creation scope.
     *
     * For example:
     * ```
     *   // Changing graph, name scope, and device to use for new ops.
     *   createWith(graph = g, nameScope = "Nested", device = "/GPU:0") {
     *     val c = constant(11.0, name = "C")
     *     assert(c.graph == g)
     *     assert(c.op.name == "Nested/C")
     *     assert(c.device == "/device:GPU:0")
     *   }
     * ```
     *
     * @param  graph               Graph to use as default for new ops.
     * @param  nameScope           Name scope to use.
     * @param  device              Device to use.
     * @param  deviceFunction      Device function to use.
     * @param  colocationOps       Colocation ops to use.
     * @param  controlDependencies Control dependencies to use.
     * @param  attributes          Attributes to use.
     * @param  container           Container to use for resources.
     * @param  block               Code block to run using the provided options.
     * @tparam R Return type of the code block.
     * @return Return value of the code block.
     */
    fun <R> with(graph: Graph? = null, nameScope: String? = null,
                 device: String = "", deviceFunction: DeviceFunction = { it.device },
                 colocationsOps: Set<Op>? = null,
                 controlDependencies: Set<Op>? = null,
                 attributes: Map<String, tensorflow.AttrValue>? = null,
                 container: String? = null,
                 block: () -> R): R {
      // TODO: Move this to a separate scope class.
      // TODO: !!! The order of the updates matters here so let's make sure everything is fine.
      var updatedContext = graphConstructionScope.value
      val newGraph = mergeGraph(graph, updatedContext)
      updatedContext = updatedContext.copy(graph = newGraph, outerContext = updatedContext)
      val newNameScope = mergeNameScope(nameScope, updatedContext.nameScope) { updatedContext.graph.uniqueName(it) }
      updatedContext = updatedContext.copy(nameScope = newNameScope, outerContext = updatedContext)
      val newDevice = mergeDevice(device, updatedContext.device)
      updatedContext = updatedContext.copy(device = newDevice, outerContext = updatedContext)
      val newDeviceFn = mergeDeviceFunction(deviceFunction, updatedContext.deviceFunction, updatedContext.device)
      updatedContext = updatedContext.copy(deviceFunction = newDeviceFn, outerContext = updatedContext)
      val newColocationsOps = mergeColocationOps(colocationsOps, updatedContext)
      updatedContext = updatedContext.copy(colocationOps = newColocationsOps, outerContext = updatedContext)
      val (newControlDependencies, newCOntrolFlowContext) = mergeControlDependencies(controlDependencies, updatedContext)
      updatedContext = updatedContext.copy(controlDependencies = newControlDependencies,
                                           controlFlowContext = newCOntrolFlowContext,
                                           outerContext = updatedContext)
      val newAttributes = mergeAttributes(attributes, updatedContext)
      updatedContext = updatedContext.copy(attributes = newAttributes, outerContext = updatedContext)
      val newContainer = mergeContainer(container, updatedContext)
      updatedContext = updatedContext.copy(container = newContainer, outerContext = updatedContext)
      return graphConstructionScope.with(updatedContext) {
        block()
      }
    }
    
    /**
     * 在[ctxNs]下新生成名称为[nameScope]的sub[NameScope]（名称冲突则会重新命名解决冲突），
     * 并将其赋值到[ctxNs]。
     *
     * [block]执行结束后，恢复[ctxNs]为调用[name_scope]之前的[NameScope]
     */
    fun <R> name_scope(nameScope: String, values: Set<Op> = emptySet(), block: () -> R): R {
      val scope = graphConstructionScope
      return if (values.isNotEmpty()) {
        val newGraph = mergeGraph(getGraphFromInputs(values), scope.value)
        val newNameScope = mergeNameScope(nameScope, scope.value.nameScope) { newGraph.uniqueName(it) }
        scope.with(scope.value.copy(graph = newGraph, nameScope = newNameScope, outerContext = scope.value)) {
          block()
        }
      } else {
        val newNameScope = mergeNameScope(nameScope, scope.value.nameScope) { scope.value.graph.uniqueName(it) }
        scope.with(scope.value.copy(nameScope = newNameScope, outerContext = scope.value)) {
          block()
        }
      }
    }
    
    /** Merges a device to the provided op creation context device and returns the device to use when specifying the
     * updated op creation context. The merging rules are specified in the documentation of the [[createWith]] function.
     *
     * @param  device    Device to merge.
     * @param  oldDevice Old (i.e., current) device.
     * @return Device to use for the new op creation context.
     */
    private fun mergeDevice(device: String, oldDevice: String): String {
      // Check if the device has been reset or has to be reset for all subsequent nested scopes
      return if (oldDevice.isEmpty() || device.isEmpty())
        ""
      else {
        val oldDeviceSpec = DeviceSpecification.fromString(oldDevice)
        val newDeviceSpec = DeviceSpecification.fromString(device)
        DeviceSpecification.merge(oldDeviceSpec, newDeviceSpec).toString()
      }
    }
    
    /** Merges a device function to the provided op creation context device and returns the device to use when specifying
     * the updated op creation context. The merging rules are specified in the documentation of the [[createWith]]
     * function.
     *
     * @param  deviceFunction    Device function to merge.
     * @param  oldDeviceFunction Old (i.e., current) device function.
     * @param  oldDevice         Old (i.e., current) device.
     * @return Device function to use for the new op creation context.
     */
    private fun mergeDeviceFunction(deviceFn: DeviceFunction, oldDeviceFn: DeviceFunction, oldDevice: String): DeviceFunction = {
      val oldDeviceSpecString = oldDeviceFn(it)
      val newDeviceSpecString = deviceFn(it)
      // Check if the device has been reset or has to be reset for all subsequent nested scopes
      if (oldDevice.isEmpty() || oldDeviceSpecString.isEmpty() || newDeviceSpecString.isEmpty())
        ""
      else {
        val oldDeviceSpec = DeviceSpecification.fromString(oldDeviceSpecString)
        val newDeviceSpec = DeviceSpecification.fromString(newDeviceSpecString)
        DeviceSpecification.merge(oldDeviceSpec, newDeviceSpec).toString()
      }
    }
    
    /** Merges a set of colocation ops to the provided op creation context set of colocation ops and returns the
     * set of colocation ops to use when specifying the updated op creation context. The merging rules are
     * specified in the documentation of the [[createWith]] function.
     *
     * @param  colocationOps Set of colocation ops to merge.
     * @param  context       Op creation context whose colocation ops need to be updated.
     * @return Set of colocation ops to use for the new op creation context.
     */
    private fun mergeColocationOps(colocationsOps: Set<Op>?, context: GraphConstructionScope): MutableSet<Op> =
        when {
          colocationsOps == null -> context.colocationOps
          colocationsOps.isEmpty() -> mutableSetOf()
          else -> {
            context.colocationOps.addAll(colocationsOps)
            context.colocationOps
          }
        }
    
    /** Merges a set of control dependencies to the provided op creation context set of control dependencies and returns
     * the set of control dependencies to use when specifying the updated op creation context. The merging rules are
     * specified in the documentation of the [with] function.
     *
     * @param  controlDependencies Set of control dependencies to merge.
     * @param  context             Op creation context whose control dependencies needs to be updated.
     * @return Set of control dependencies to use for the new op creation context.
     */
    private fun mergeControlDependencies(controlDependencies: Set<Op>?, context: GraphConstructionScope)
        : Pair<MutableSet<Op>, ControlFlowContext?> =
        when {
          controlDependencies == null -> context.controlDependencies to context.controlFlowContext
          controlDependencies.isEmpty() -> mutableSetOf<Op>() to null
          else -> {
            context.controlDependencies.addAll(controlDependencies)
            context.controlDependencies to context.controlFlowContext
          }
        }
    
    /** Merges a set of attributes to the provided op creation context set of attributes and returns the set of attributes
     * to use when specifying the updated op creation context. The merging rules are specified in the documentation of
     * the [[createWith]] function.
     *
     * @param  attributes Set of attributes to merge.
     * @param  context    Op creation context whose attributes needs to be updated.
     * @return Set of attributes to use for the new op creation context.
     */
    private fun mergeAttributes(attributes: Map<String, tensorflow.AttrValue?>?, context: GraphConstructionScope): MutableMap<String, tensorflow.AttrValue> =
        when {
          attributes == null -> context.attributes
          attributes.isEmpty() ->
            mutableMapOf()
          else -> {
            val mergedMap = context.attributes
            attributes.forEach { (key, value) ->
              if (value == null && mergedMap.contains(key))
                mergedMap.remove(key)
              else if (value != null)
                mergedMap[key] = value
            }
            mergedMap
          }
        }
    
    /** Merges a container to the provided op creation context container and returns the container to use when specifying
     * the updated op creation context. The merging rules are specified in the documentation of the [[createWith]]
     * function.
     *
     * @param  container Container to merge.
     * @param  context   Op creation context whose container needs to be updated.
     * @return Container to use for the new op creation context.
     */
    private fun mergeContainer(container: String?, context: GraphConstructionScope): String =
        container ?: context.container
    
    /** Merges a graph to the provided op creation context graph and returns the graph to use when specifying the updated
     * op creation context. The merging rules are specified in the documentation of the [with] function.
     *
     * @param  graph   Graph to merge.
     * @param  context Op creation context whose graph needs to be updated.
     * @return Graph to use for the new op creation context.
     */
    private fun mergeGraph(graph: Graph?, context: GraphConstructionScope) =
        graph ?: context.graph
    
    /** Merges a name scope to the provided op creation context name scope and returns the name scope to use when
     * specifying the updated op creation context. The merging rules are specified in the documentation of the
     * [[createWith]] function.
     *
     * @param  nameScope    Name scope to merge.
     * @param  oldNameScope Old (i.e., current) name scope.
     * @param  uniqueNameFn Function that can be used to generate a unique name based on a provided name.
     * @return Name scope to use for the new op creation context.
     * @throws IllegalNameException If the provided name scope does not pass the regular expression validity checks.
     */
    private fun mergeNameScope(nameScope: String?, oldNameScope: String, uniqueNameFn: (String) -> String): String {
      return if (nameScope == null)
        oldNameScope
      else {
        // Check whether the provided name scope is valid.
        // If the root name scope is being set, then stricter checks are performed on it (i.e., op naming checks). This
        // makes sure the name scope does not start with any illegal characters (e.g., '_', '-', '\', and '/').
        if ((oldNameScope == "" && nameScope != "" && !checkName(nameScope))
            || (oldNameScope != "" && !checkNameScope(nameScope)))
          throw IllegalNameException("Illegal name scope '$nameScope'.")
        when {
          nameScope == "" -> ""
          nameScope.endsWith('/') -> convertNameScopeToName(nameScope)
          else -> uniqueNameFn(nameScope)
        }
      }
    }
    
    /** Returns the appropriate graph to use for the given inputs.
     *
     * This function provides a consistent algorithm for choosing the graph in which an op should be constructed in:
     *
     *   1. If the argument `graph` is provided and is not set to `null`, the function validates that all `inputs` are
     * defined in that graph.
     *   2. Otherwise, we attempt to select a graph from the first op in `inputs` and validate that all other `inputs`
     * are also defined in the same graph.
     *
     * @param  inputs Inputs.
     * @param  graph  Graph to use. If `null`, the graph is inferred from `inputs`.
     * @return The appropriate graph to use for the given inputs.
     * @throws GraphMismatchException If any two of the inputs lie in different graphs, or if `graph` is not `null` and
     *                                at least one of the `inputs` is not defined in it.
     * @see "tensorflow.python.framework.ops._get_graph_from_inputs"
     *
     */
    private fun getGraphFromInputs(inputs: Set<Op>, graph: Graph? = null): Graph {
      val returnGraph = graph ?: inputs.first().graph
      inputs.forEach {
        if (graph == null)
          assetSameGraph(inputs.first(), it)
        else if (it.graph != returnGraph)
          throw GraphMismatchException("'$it' is not defined in the passed-in graph.")
      }
      return returnGraph
    }
    
    /** Asserts that two ops are defined in the same graph. If they are not, a [[GraphMismatchException]] is thrown.
     *
     * @param  op1 First op.
     * @param  op2 Second op.
     * @throws GraphMismatchException If the two ops lie in different graphs.
     * @see "tensorflow.python.framework.ops._assert_same_graph"
     */
    private fun assetSameGraph(op1: Op, op2: Op) {
      if (op1.graph != op2.graph)
        throw GraphMismatchException("$op1 must be from the same graph as $op2")
    }
    
    fun <R> attr_scope(vararg attrs: Pair<String, AttrValue>, block: () -> R): R =
        with(attributes = attrs.associate { it }, block = block)
    
    fun debugString() = GraphDef.parseFrom(currentGraph.toGraphDef()).toString()
    fun printGraph() {
      debugString().println()
    }
    
    fun session(block: Session.() -> Unit) {
      val session = Session(currentGraph.c_graph)
      block(session)
    }
    
    fun global_variable_initializer(): Op {
      TODO()
//    return tf.group(global_variables, name = "init")
    }
    
    val version get() = TF_Version().string
    
    fun <R> condContext(pred: Output, pivot: Output, branch: Int, block: (CondContext) -> R): R {
//    val tmp = currentControlFlowContext
//    currentControlFlowContext = CondContext(predicate, pivot, branch)
//    try {
//      return block(currentControlFlowContext as CondContext)
//    } finally {
//      currentControlFlowContext = tmp
//    }
      TODO()
    }
    
    fun <R> on_device(dev: String, block: () -> R): R =
        with(device = dev, block = block)
    
    fun <R> on_device(dev: DeviceFunction, block: () -> R): R =
        with(deviceFunction = dev, block = block)
    
    fun <R> control_dependencies(control_inputs: Output, block: () -> R) =
        control_dependencies(control_inputs.op!!) { block() }
    
    fun <R> colocate_with(op: Output, ignore_existing: Boolean = false, block: () -> R) =
        colocate_with(op.op!!, ignore_existing, block)
    
    fun <R> colocate_with(op: Op, ignore_existing: Boolean = false, block: () -> R) =
        colocate_with(setOf(op), ignore_existing, block)
    
    fun <R> colocate_with(ops: Set<Op>, ignore_existing: Boolean = false, block: () -> R): R =
        with(colocationsOps = ops, block = block)
    
    fun <R> colocate_with_tensors(op: Array<Output>, ignore_existing: Boolean = false, block: () -> R): R {
      return colocate_with(op.mapTo(mutableSetOf<Op>()) { it.op!! }, ignore_existing, block)
    }
    
    fun <R> colocate_with_tensors(op: List<Output>, ignore_existing: Boolean = false, block: () -> R): R {
      return colocate_with(op.mapTo(mutableSetOf<Op>()) { it.op!! }, ignore_existing, block)
    }
    
    fun <R> control_dependencies(vararg control_inputs: Op, block: () -> R): R =
        with(controlDependencies = setOf(*control_inputs), block = block)
    
    fun <R> control_dependencies(control_inputs: List<Op>, block: () -> R): R =
        with(controlDependencies = control_inputs.toSet(), block = block)
    
    //    private var tmpDev = ""
    fun begin_device(s: String) {
      TODO("not implemented")
    }
    
    fun end_device() {
      TODO("not implemented")
    }
  }
  
}