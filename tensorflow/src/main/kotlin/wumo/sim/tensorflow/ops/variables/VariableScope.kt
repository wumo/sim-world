package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.core.Graph.Graph
import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.tensorflow.core.ShapeMismatchException
import wumo.sim.tensorflow.ops.DeviceFunction
import wumo.sim.tensorflow.ops.variables.Variable.VariableGetter
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.types
import wumo.sim.util.Shape
import wumo.sim.util.emptyMutableSet

/** Variable scope that carries default settings to provide to [getVariable].
 *
 * A variable scope allows to create new variables and to share already created ones while providing checks to not
 * create or share by accident.
 *
 * Many of the arguments we need for [getVariable] in a variable store are most easily handled with a context.
 * [VariableScope] objects are used for the defaults.
 *
 * @param  reuse            [Reuse] value indicating whether to re-use an existing variable with the same name,
 *                          create a new variable, or do either.
 * @param  name             Name of the variable scope, used as a prefix in `getVariable`.
 * @param  initializer      Default initializer passed to `getVariable`.
 * @param  regularizer      Default regularizer passed to `getVariable`.
 * @param  partitioner      Default partitioner passed to `getVariable`.
 * @param  cachingDevice    Default caching device passed to `getVariable`.
 * @param  nameScope        Default name scope passed to `getVariable`.
 * @param  dataType         Default data type passed to `getVariable`.
 * @param  underlyingGetter Default underlying variable getter passed to `getVariable`.
 *
 * 此[VariableScope]对应的[NameScope]，通常是复用的
 */
internal class VariableScope(
    val reuse: Reuse,
    val name: String? = "",
    val dataType: DataType<*>? = types.FLOAT16,
    val initializer: Initializer? = null,
    val regularizer: Regularizer? = null,
    val cachingDevice: DeviceFunction? = null,
    val partitioner: Partitioner? = null,
    val nameScope:String="",
    val underlyingGetter: VariableGetter? = null) {
  
  /** Gets an existing variable with the specified name or creates a new one.
   *
   * @param  store         Variable store currently being used to store variables.
   * @param  name          Variable name.
   * @param  dataType      Variable data type.
   * @param  shape         Variable shape.
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
   *                       - If `reuse` is `null` (the default), both new and existing variables are returned.
   * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
   *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
   * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
   *                       to the variable's device. Typical use is to cache on the device where the ops using the
   *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
   * @return Requested variable.
   * @throws IllegalArgumentException If any of the provided arguments are not compatible with each other, or with the
   *                                  variables stored in this variable store.
   * @throws ShapeMismatchException   If the provided shape does not match the shape of the corresponding variable
   *                                  stored in this variable store (if there exists one).
   * @throws InvalidDataTypeException If the provided data type does not match the data type of the corresponding
   *                                  variable stored in this variable store (if there exists one).
   */
  fun getVariable(
      store: VariableStore,
      name: String,
      shape: Shape? = null,
      dataType: DataType<*> = FLOAT,
      initializer: Initializer? = null,
      regularizer: Regularizer? = null,
      trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew,
      collections: MutableSet<Graph.Key<Variable>> = emptyMutableSet(),
      cachingDevice: DeviceFunction? = null
  ): Variable {
    val fullName = if (this.name != null && this.name != "") "${this.name}/$name" else name
    // Variable names only depend on the variable scope and not the name scope,
    // so we reset it below for the time of variable creation.
    return tf.name_scope("") {
      store.getVariable(fullName, shape, dataType, initializer, regularizer, trainable, reuse, collections, cachingDevice)
    }
  }
  
  companion object {
    /** Returns the current variable scope. */
    val current: VariableScope
      get() = VariableScopeStore.current.scope
    
    /** Sets the variable scope to use for op creation context, for all code in `block`.
     *
     * @param  name             Variable scope name, that may also change the name scope of the op creation context,
     *                          depending on the value of [isPure].
     * @param  reuse            [Reuse] value indicating whether to re-use an existing variable with the same name, or
     * do either. Note that this argument cannot be set to [CreateNewOnly] in this function.If set to [[ReuseOrCreateNew]],
     * then the parent variable scope `reuse` value is used (i.e., propagated).
     * @param  dataType         Default data type for variables within the scope.
     * @param  initializer      Default initializer for variables within the scope.
     * @param  regularizer      Default regularizer for variables within the scope.
     * @param  partitioner      Default partitioner for variables within the scope.
     * @param  cachingDevice    Default caching device for variables within the scope.
     * @param  underlyingGetter Default variable getter for variables within the scope.
     * @param  isDefaultName    Boolean value indicating whether `name` is a default name or not. If `true`, then `name`
     *                          will be made unique before being used. `isDefaultName` cannot be set to `true` when
     *                          `reuse` is set to [[ReuseExistingOnly]].
     * @param  isPure           Boolean value indicating whether to use a "pure" variable scope. That is, a variable
     *                          scope that does not affect the name scope of the current op creation context.
     * @param  block            Code block to run using the provided options.
     * @return Return value of the code block.
     */
    internal fun <R> scope(
        name: String,
        reuse: Reuse = ReuseOrCreateNew,
        dataType: DataType<*>? = null,
        initializer: Initializer? = null,
        regularizer: Regularizer? = null,
        cachingDevice: DeviceFunction? = null,
        partitioner: Partitioner? = null,
        underlyingGetter: VariableGetter? = null,
        isDefaultName: Boolean = false,
        isPure: Boolean = false,
        block: () -> R
    ): R {
      val variableScopeStore = VariableScopeStore.current
      val oldVariableScope = variableScopeStore.scope
      val newName = run {
        val uniqueName = if (isDefaultName) name else name
        if (oldVariableScope.name != null && oldVariableScope.name != "")
          "${oldVariableScope.name}/$uniqueName"
        else
          uniqueName
      }
      variableScopeStore.enterVariableScope(newName)
      
      val newVariableScope = VariableScope(
          reuse = if (reuse == ReuseOrCreateNew) oldVariableScope.reuse else reuse,
          name = newName,
          dataType = dataType ?: oldVariableScope.dataType,
          initializer = initializer ?: oldVariableScope.initializer,
          regularizer = regularizer ?: oldVariableScope.regularizer,
          cachingDevice = cachingDevice ?: oldVariableScope.cachingDevice,
          partitioner = partitioner ?: oldVariableScope.partitioner,
          nameScope = name,
          underlyingGetter = if (underlyingGetter == null) oldVariableScope.underlyingGetter
          else maybeWrapCustomVariableGetter(underlyingGetter, oldVariableScope.underlyingGetter)
      )
      variableScopeStore.scope = newVariableScope
      val result = if (isPure) block() else tf.name_scope(name) { block() }
      variableScopeStore.closeVariableSubScopes(newName)
      variableScopeStore.scope = oldVariableScope
      return result
    }
    
    /**
     * If a new getter is provided, it wraps around the old one and the new wrapped getter is returned. Otherwise, the
     * old getter is returned.
     * @see "tensorflow.python.ops.variable_scope._maybe_wrap_custom_getter"
     */
    private fun maybeWrapCustomVariableGetter(getter: VariableGetter, oldGetter: VariableGetter?): VariableGetter {
      if (oldGetter == null)
        return getter
      
      return object : VariableGetter {
        override fun invoke(
            name: String,
            dataType: DataType<*>,
            shape: Shape?,
            initializer: Initializer?,
            regularizer: Regularizer?,
            trainable: Boolean, reuse: Reuse,
            collections: MutableSet<Graph.Key<Variable>>,
            cachingDevice: DeviceFunction?,
            underlyingGetter: VariableGetter?): Variable {
          val baseGetter = object : VariableGetter by oldGetter {}
          return getter(name, dataType, shape, initializer, regularizer,
                        trainable, reuse, collections, cachingDevice, baseGetter)
        }
      }
    }
  }

//  override fun enter() {
//    var_scope_count.clear()
//  }
//
//  override fun exit() {
//    var_scope_count.clear()
//  }
//
//  /**是否允许重复进入[VariableScope]时自动递增后缀*/
//  var reenter_increment = false
//  /***/
//  private val var_scope_count = HashMap<String, Int>()
//  /**这一层存储的所有变量*/
//  val variables = LinkedHashMap<String, Variable>()
//  /**这一层下的sub variable scope, */
//  val variable_subscopes = LinkedHashMap<String, VariableScope>()
//
//  fun all_variables(): List<Variable> {
//    val a_variables = mutableListOf<Variable>()
//    a_variables += variables.values
//    for (variableScope in variable_subscopes.values) {
//      a_variables += variableScope.all_variables()
//    }
//    return a_variables
//  }
//
//  /**
//   * 在[enter]后，[exit]之前，调用[variable_scope]进入
//   * 某个[name]的sub[VariableScope]时，第一次进入时复用已有的
//   * [VariableScope]，之后再次进入重复[name]的sub[VariableScope]时，
//   * 则自动递增[name]的后缀,递增条件如下：
//   *
//   * 1. 如[name]="a"，而之前为访问过，则返回"a"，记录访问过1次;
//   * 2. 如果"a"之前访问过1次，则更改为"a_1"，记录访问过2次；
//   * 3. 如果"a"之前访问过2此，则更改为"a_2"，记录访问过3次
//   *
//   * [exit]之后，计数清空，重新开始
//   */
//  fun variable_scope(name: String, reuse: Boolean, reenter_increment: Boolean): VariableScope {
//    assert(name.isNotEmpty())
//    val suffix =
//        if (!reenter_increment) ""
//        else {
//          val visits = var_scope_count.compute(name) { _, v ->
//            val visits = (v ?: 0)
//            visits + 1
//          }!!
//          if (visits == 1) "" else "_${visits - 1}"
//        }
//    val real_name = "$name$suffix"
//    return variable_subscopes.compute(real_name) { _, v ->
//      v ?: VariableScope(real_name, nameScope.reuse_or_new_subscope(real_name))
//    }!!.apply {
//      this.reuse = reuse
//      this.reenter_increment = reenter_increment
//    }
//  }
}