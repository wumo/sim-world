package wumo.sim.tensorflow.ops.training

import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.tensorflow.ops.IndexedSlices
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike
import wumo.sim.tensorflow.ops.gradients.gradient_ops
import wumo.sim.tensorflow.ops.gradients.gradient_ops.AggregationMethod.AddAggregationMethod
import wumo.sim.tensorflow.ops.gradients.gradient_ops.GatingMethod.GraphGating
import wumo.sim.tensorflow.ops.gradients.gradient_ops.GatingMethod.OpGating
import wumo.sim.tensorflow.ops.training.Optimizer.Companion.VariableProcessor.RefVariableProcessor
import wumo.sim.tensorflow.ops.variables.DynamicInitializer
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tensor.Tensor
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.*

abstract class Optimizer {
  /** Name of this optimizer. This name is used for the accumulators created for this optimizer. */
  abstract val name: String
  
  /** Boolean value indicating whether to apply use locks to prevent concurrent updates to variables. */
  abstract val useLocking: Boolean
  
  /** Boolean value indicating whether to ignore duplicate indices during sparse updates. */
  abstract val ignoreDuplicateSparseIndices: Boolean
  
  /** Some [Optimizer] subclasses use additional variables. For example, `MomentumOptimizer` and `AdaGradOptimizer`
   * use variables to accumulate updates. This map is where these variables are stored. */
  protected val slots = mutableMapOf<String, MutableMap<Variable, Variable>>()
  
  /** Returns the names of all slots used by this optimizer. */
  protected val slotNames get() = slots.keys
  
  /**
   * Additional variables created by the `Optimizer`.
   * Contains variables used by some optimizers that require no slots to be stored.
   */
  val nonSlotVariables = mutableMapOf<String, Variable>()
  
  /** Creates an op that makes a step towards minimizing `loss` by updating the values of the variables in `variables`.
   *
   * This method simply combines calls [[computeGradients]] and [[applyGradients]]. If you want to process the
   * gradients before applying them call [[computeGradients]] and [[applyGradients]] explicitly instead of using this
   * method.
   *
   * @param  loss                       Loss value whose gradients will be computed.
   * @param  gradLoss              Optional gradients to back-propagate for `loss`.
   * @param  variables                  Optional list of variables for which to compute the gradients. Defaults to the
   *                                    set of trainable variables in the graph where `loss` is defined.
   * @param  gradientsGatingMethod      Gating method for the gradients computation.
   * @param  gradientsAggregationMethod Aggregation method used to combine gradient terms.
   * @param  colocateGradientsWithOps   Boolean value indicating whether to colocate the gradient ops with the original
   *                                    ops.
   * @param  globalStep                  Optional `Variable` to increment by one after the variables have been updated.
   * @param  name                       Name for the created op.
   * @return Created op.
   *
   * @see "tensorflow.python.training.optimizer.Optimizer#minimize"
   */
  fun minimize(
      loss: Output,
      gradLoss: List<OutputLike>? = null,
      variables: Set<Variable>? = null,
      gradientsGatingMethod: gradient_ops.GatingMethod = OpGating,
      gradientsAggregationMethod: gradient_ops.AggregationMethod = AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false,
      globalStep: Variable? = null,
      name: String = "Minimize"
  ): Op {
    val gradientsAndVariables = computeGradients(
        loss, gradLoss, variables, gradientsGatingMethod,
        gradientsAggregationMethod, colocateGradientsWithOps)
    return applyGradients(gradientsAndVariables, globalStep, name)
  }
  
  /** Computes the gradients of `loss` with respect to the variables in `variables`, if provided, otherwise with respect
   * to all the trainable variables in the graph where `loss` is defined.
   *
   * @param  loss                       Loss value whose gradients will be computed.
   * @param  gradLoss              Optional gradients to back-propagate for `loss`.
   * @param  variables                  Optional list of variables for which to compute the gradients. Defaults to the
   *                                    set of trainable variables in the graph where `loss` is defined.
   * @param  gradientsGatingMethod      Gating method for the gradients computation.
   * @param  gradientsAggregationMethod Aggregation method used to combine gradient terms.
   * @param  colocateGradientsWithOps   Boolean value indicating whether to colocate the gradient ops with the original
   *                                    ops.
   * @return Sequence of gradient-variable pairs.
   * @see "tensorflow.python.training.optimizer.Optimizer#compute_gradients"
   */
  fun computeGradients(
      loss: Output,
      gradLoss: List< OutputLike>? = null,
      variables: Set<Variable>? = null,
      gradientsGatingMethod: gradient_ops.GatingMethod = OpGating,
      gradientsAggregationMethod: gradient_ops.AggregationMethod = AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false
  ): List<Pair<OutputLike?, Variable>> {
    assertValidDataTypes(listOf(loss))
    if (gradLoss != null)
      assertValidDataTypes(gradLoss)
    val collectedVariables = (variables ?: loss.graph.trainableVariables) +
        loss.graph.getCollection(Graph.Graph.Keys.STREAMING_MODEL_PORTS)
    if (collectedVariables.isEmpty())
      throw IllegalArgumentException("There are no variables to optimize.")
    val variableProcessors = collectedVariables.map { getVariableProcessor(it) }
    val variableTargets = variableProcessors.map { it.target }
    var gradients = tf.gradients(listOf(loss),
                                 variableTargets,
                                 grad_ys = gradLoss,
                                 gateGradients = gradientsGatingMethod == OpGating,
                                 aggregationMethod = gradientsAggregationMethod,
                                 colocateGradientsWithOps = colocateGradientsWithOps)
    if (gradientsGatingMethod == GraphGating)
      gradients = tf.tuple(gradients)
    val gradientsAndVariables = gradients.zip(collectedVariables)
    assertValidDataTypes(gradientsAndVariables.asSequence()
                             .filter { (g, v) -> g != null && v.dataType != RESOURCE }
                             .map { it._2.value }
                             .toList())
    return gradientsAndVariables
  }
  
  /** Creates an op that applies the provided gradients to the provided variables.
   *
   * @param  gradientsAndVariables Sequence with gradient-variable pairs.
   * @param  global_step             Optional `Variable` to increment by one after the variables have been updated.
   * @param  name                  Name for the created op.
   * @return Created op.
   * @see "tensorflow.python.training.optimizer.Optimizer#apply_gradients"
   */
  fun applyGradients(
      gradientsAndVariables: List<Pair<OutputLike?, Variable>>,
      global_step: Variable? = null,
      name: String = this.name
  ): Op {
    // This is a default implementation of `applyGradients` that is shared by most optimizers. It relies on the subclass
    // implementing the following methods: `createSlots`, `prepare`, `finish`, `applyDense`, and `applySparse`.
    val gradientsAndVariables = gradientsAndVariables.asSequence()
        .filter { it._1 != null }
        .map { t3(it._1, it._2, getVariableProcessor(it._2)) }
        .toList()
    val variables = gradientsAndVariables.map { it._2 }
    if (variables.isEmpty())
      throw IllegalArgumentException("No gradients were provided for any of the variables: " +
                                         "${gradientsAndVariables.map { it._2 }.joinToString(", ")}.")
    tf.init_scope {
      createSlots(variables)
    }
    
    return tf.nameScope(name) {
      prepare(global_step)
      val updateOps = mutableSetOf<Op>()
      for ((g, v, p) in gradientsAndVariables)
      // We colocate all ops created for variable application on the same device as the variable.
        tf.with(nameScope = "update_${v.op.name}", colocationsOps = mutableSetOf(v.op)) {
          updateOps += p.updateOp(this, g!!, global_step)
        }
      
      val apply_updates = if (global_step == null)
        finish(updateOps, tf.currentNameScope)
      else
        tf.controlDependencies(mutableSetOf(finish(updateOps, "update"))) {
          tf.colocateWith(mutableSetOf(global_step.op)) {
            // The implicit read in the default assign add operation in `Variable` is slow and so we avoid that here.
            global_step.assignAdd(tf.const(global_step.dataType, 1), tf.currentNameScope).op
          }
        }
      updateOps.first().graph.addToCollection(apply_updates, Graph.Graph.Keys.TRAIN_OP)
      apply_updates
    }
  }
  
  /** Supported data types for the loss function, the variables, and the gradients. Subclasses should override this
   * field allow other float types. */
  private val validDataTypes = setOf(FLOAT16, BFLOAT16, FLOAT, DOUBLE)
  
  /** Asserts that the provided `outputs` all have data types that are supported by this optimizer.
   *
   * @param  outputs Outputs whose data types to check.
   * @throws InvalidDataTypeException If any of the provided outputs has an unsupported data type.
   */
  private fun assertValidDataTypes(outputs: Iterable<OutputLike>) {
    outputs.forEach {
      if (it.dataType !in validDataTypes)
        throw InvalidDataTypeException("Data type '${it.dataType}' is not supported by this optimizer.")
    }
  }
  
  /** Create all slots needed by this optimizer. */
  open fun createSlots(variables: List<Variable>) {
    // No slots are created by default.
  }
  
  /** Creates all necessary tensors before applying the gradients. This function is called from within an op creation
   * context that uses as its name native the name that users have chosen for the application of gradients. */
  open fun prepare(iteration: Variable?) {}
  
  /** Creates an op that finishes the gradients application. This function is called from within an op creation context
   * that uses as its name native the name that users have chosen for the application of gradients.
   *
   * @param  updateOps Set of ops needed to apply the gradients and update the variable values.
   * @param  nameScope Name native to use for all the ops created by this function.
   * @return Created op output.
   */
  open fun finish(updateOps: Set<Op>, nameScope: String): Op =
      tf.group(updateOps, nameScope)
  
  /** Applies the updates corresponding to the provided gradient, to the provided variable.
   *
   * @param  gradient  Gradient tensor.
   * @param  variable  Variable.
   * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
   * @return Created op that applies the provided gradient to the provided variable.
   */
  abstract fun applyDense(gradient: Output, variable: Variable, iteration: Variable?): Op
  
  /** Applies the updates corresponding to the provided gradient, to the provided variable.
   *
   * The [[OutputIndexedSlices]] object specified by `gradient` in this function is by default pre-processed in
   * `applySparseDuplicateIndices` to remove duplicate indices (refer to that function's documentation for details).
   * Optimizers which can tolerate or have correct special cases for duplicate sparse indices may override
   * `applySparseDuplicateIndices` instead of this function, avoiding that overhead.
   *
   * @param  gradient  Gradient tensor.
   * @param  variable  Variable.
   * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
   * @return Created op that applies the provided gradient to the provided variable.
   */
  open fun applySparse(gradient: IndexedSlices, variable: Variable, iteration: Variable?): Op =
      NONE()
  
  /** Applies the updates corresponding to the provided gradient (with potentially duplicate indices), to the provided
   * variable.
   *
   * Optimizers which override this method must deal with [[OutputIndexedSlices]] objects such as the following:
   * `OutputIndexedSlices(indices=[0, 0], values=[1, 1], denseShape=[1])`, which contain duplicate indices. The
   * correct interpretation in that case should be: `OutputIndexedSlices(values=[2], indices=[0], denseShape=[1])`.
   *
   * Many optimizers deal incorrectly with repeated indices when updating based on sparse gradients (e.g. summing
   * squares rather than squaring the sum, or applying momentum terms multiple times). Adding first is always the
   * correct behavior, so this is enforced here by reconstructing the [[OutputIndexedSlices]] to have only unique
   * indices, and then calling [[applySparse]].
   *
   * Optimizers which deal correctly with repeated indices may instead override this method to avoid the induced
   * overhead.
   *
   * @param  gradient  Gradient tensor.
   * @param  variable  Variable.
   * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
   * @return Created op that applies the provided gradient to the provided variable.
   */
  open fun applySparseDuplicateIndices(
      gradient: IndexedSlices,
      variable: Variable,
      iteration: Variable?
  ): Op = NONE()
  
  /** Gets the map used for caching slots created under the provided name. If the map does not exist, then a new empty
   * map is created and returned.
   *
   * @param  name Slot name.
   * @return Map used for caching slots created under the provided name.
   */
  private fun slotMap(name: String): MutableMap<Variable, Variable> =
      slots.getOrPut(name) { mutableMapOf() }
  
  /** Gets an existing slot or creates a new one if none exists, for the provided arguments.
   *
   * @param  name          Slot name.
   * @param  variable      Slot primary variable.
   * @param  initializer   Slot variable initializer.
   * @param  shape         Slot variable shape.
   * @param  dataType      Slot variable data type.
   * @param  variableScope Name to use when scoping the variable that needs to be created for the slot.
   * @return Requested slot variable.
   */
  protected fun getSlot(
      name: String,
      variable: Variable,
      initializer: Initializer,
      shape: Shape,
      dataType: DataType<*>,
      variableScope: String
  ): Variable = tf.colocateWith(mutableSetOf(variable.op)) {
    slotMap(name).getOrPut(variable) {
      slot_creator.create(variable, initializer, variableScope, dataType, shape)
    }
  }
  
  /** Return a slot named `name` created for `var` by the Optimizer.
   *
   * @param  name     Slot name.
   * @param  variable Slot primary variable.
   * @return Requested slot variable, or `null` if it cannot be found.
   */
  protected fun getSlot(name: String, variable: Variable): Variable? =
      slots.getOrDefault(name, emptyMutableMap()).get(variable)
  
  /** Gets an existing slot or creates a new one using an initial value of zeros, if none exists.
   *
   * @param  name          Slot name.
   * @param  variable      Slot primary variable.
   * @param  variableScope Name to use when scoping the variable that needs to be created for the slot.
   * @return Requested slot variable.
   */
  protected fun zerosSlot(name: String, variable: Variable, variableScope: String): Variable {
    return tf.colocateWith(mutableSetOf(variable.op)) {
      slotMap(name).getOrPut(variable) {
        slot_creator.zeros(variable, variableScope)
      }
    }
  }
  
  /** Gets or creates (and adds to this optimizer) a non-slot variable.
   *
   * @param  name          Variable name.
   * @param  initialValue  Variable initial value.
   * @param  colocationOps Set of colocation ops for the non-slot variable.
   * @return Created non-slot variable.
   */
  protected fun getOrCreateNonSlotVariable(
      name: String,
      initialValue: Tensor<*>,
      colocationOps: MutableSet<Op> = mutableSetOf()
  ): Variable =
      nonSlotVariables.getOrPut(name) {
        tf.colocateWith(colocationOps) {
          Variable.getVariable(name,
                               initializer = DynamicInitializer(tf.const(initialValue)),
                               trainable = false)
        }
      }
  
  /** Gets a non-slot variable that has been added to this optimizer (or throws an error if no such non-slot variable
   * could be found in this optimizer).
   *
   * @param  name  Variable name.
   * @param  graph Graph in which the variable is defined.
   * @return Obtained non-slot variable.
   */
  protected fun getNonSlotVariable(name: String, graph: Graph? = null): Variable? =
      nonSlotVariables[name]
  
  /** Gets all the non-slot variables that have been added to this optimizer. */
  protected val getNonSlotVariables: Iterable<Variable> get() = nonSlotVariables.values
  
  /** Returns a sequence of variables which encode the current state of this optimizer. The returned variables include
   * both slot variables and non-slot global variables created by this optimizer, in the current graph. */
  val variables: List<Variable> =
      (getNonSlotVariables.filter { it.graph == tf.currentGraph } +
          slots.values.flatMap { it.values })
          .toList()
          .sortedBy { it.name }
  
  companion object {
    /**
     * @see "tensorflow.python.training.optimizer._OptimizableVariable"
     */
    sealed class VariableProcessor {
      
      /** Returns the optimization target for this variable. */
      abstract val target: Output
      
      /** Returns the update ops for updating this variable using the gradient provided by `gradient`. */
      abstract fun updateOp(optimizer: Optimizer, gradient: OutputLike, iteration: Variable?): Op
      
      /**
       * Processor for Variable.
       * @see "tensorflow.python.training.optimizer._RefVariableProcessor"
       */
      class RefVariableProcessor(val v: Variable) : VariableProcessor() {
        
        override val target: Output = v.variable
        
        override fun updateOp(optimizer: Optimizer, gradient: OutputLike, iteration: Variable?): Op =
            when (gradient) {
              is Output -> optimizer.applyDense(gradient, v, iteration)
              is IndexedSlices -> optimizer.applySparseDuplicateIndices(gradient, v, iteration)
              else -> throw IllegalArgumentException("Unsupported gradient type. Currently only 'Output' " +
                                                         "and 'IndexedSlices' are supported.")
            }
        
      }
      
      /**
       * Processor for dense ResourceVariables.
       * @see "tensorflow.python.training.optimizer._DenseReadResourceVariableProcessor"
       */
      class DenseReadResourceVariableProcessor(v: Variable) : VariableProcessor() {
        
        override val target: Output
          get() = TODO("not implemented")
        
        override fun updateOp(optimizer: Optimizer, gradient: OutputLike, iteration: Variable?): Op {
          TODO("not implemented")
        }
      }
      
      /**
       * Processor for dense ResourceVariables.
       * @see "tensorflow.python.training.optimizer._DenseResourceVariableProcessor"
       */
      class DenseResourceVariableProcessor(v: Variable) : VariableProcessor() {
        
        override val target: Output
          get() = TODO("not implemented")
        
        override fun updateOp(optimizer: Optimizer, gradient: OutputLike, iteration: Variable?): Op {
          TODO("not implemented")
        }
        
      }
      
    }
    
    internal fun getVariableProcessor(variable: Variable): VariableProcessor =
        RefVariableProcessor(variable)
  }
}

//
//import wumo.sim.tensorflow.ops.Op
//import wumo.sim.tensorflow.ops.Output
//import wumo.sim.tensorflow.ops.ops
//import wumo.sim.tensorflow.ops.variables.Variable
//import wumo.sim.tensorflow.tf
//import wumo.sim.util.t2
//
///**
// * This class defines the API to add Ops to train a model.  You never use this
// * class directly, but instead instantiate one of its subclasses such as
// * **GradientDescentOptimizer**, **AdagradOptimizer**, or **MomentumOptimizer**.
// */
//abstract class Optimizer(val use_locking: Boolean, val name: String) {
//
//  val slots = mutableMapOf<String, MutableMap<Variable, Variable>>()
//  val non_slot_dict = mutableMapOf<String, Variable>()
//
//  fun minimize(loss: Output, var_list: Collection<Variable>? = null, name: String = ""): Op {
//    val grads_and_vars = compute_gradients(loss, var_list)
//    val vars_with_grad = grads_and_vars.map { (g, v) -> v }
//    return apply_gradients(grads_and_vars, name = name)
//  }
//
//  fun compute_gradients(loss: Output, var_list: Collection<Variable>?): List<t2<Output, Variable>> {
//    val var_list = var_list ?: tf.trainables
//    val grads = tf.gradients(loss, var_list)
//    return grads.zip(var_list)
//  }
//
//  fun apply_gradients(grads_and_vars: List<t2<Output, Variable>>, name: String = ""): Op {
//    val name = if (name.isEmpty()) this.name else name
//    val var_list = grads_and_vars.map { (g, v) -> v }
//    create_slots(var_list)
//    val update_ops = mutableListOf<Op>()
//    ops.nameScope(name) {
//      prepare()
//      for ((grad, v) in grads_and_vars)
//        ops.nameScope("update_$name") {
//          ops.colocateWith(v.op!!) {
//            update_ops += apply_dense(grad, v)
//          }
//        }
//      val apply_updates = finish(update_ops, name)
//      tf.train_ops += apply_updates
//      return apply_updates
//    }
//  }
//
//  open fun create_slots(var_list: List<Variable>) {}
//
//  abstract fun prepare()
//
//  //TODO sparse IndexedSlices
//  abstract fun apply_dense(grad: Output, v: Variable): Op
//
//  open fun finish(update_ops: MutableList<Op>, name: String) =
//      tf.group(update_ops, name)
//
//  open fun slot_dict(slot_name: String): MutableMap<Variable, Variable> {
//    return slots.compute(slot_name) { _, named_slots ->
//      named_slots ?: mutableMapOf()
//    }!!
//  }
//
//  fun zero_slot(v: Variable, slot_name: String, op_name: String): Variable {
//    val named_slots = slot_dict(slot_name)
//    return named_slots.compute(v) { _, slot_variable ->
//      slot_variable ?: create_zeros_slot(v, op_name)
//
//    }!!
//  }
//
//  /**
//   * Return a slot named `name` created for `var` by the Optimizer.
//
//  Some `Optimizer` subclasses use additional variables.  For example
//  `Momentum` and `Adagrad` use variables to accumulate updates.  This method
//  gives access to these `Variable` objects if for some reason you need them.
//
//  Use `get_slot_names()` to get the list of slot names created by the
//  `Optimizer`.
//
//   * @param v: A variable passed to `minimize()` or `apply_gradients()`.
//   * @param name: A string.
//
//   * @return: The `Variable` for the slot if it was created, `None` otherwise.
//   */
//  open fun get_slot(v: Variable, name: String): Variable {
//    val named_slots = slots[name]
//    return named_slots!![v]!!
//  }
//
//  /**Add an extra variable, not associated with a slot.*/
//  open fun create_non_slot_variable(initial_value: Any, name: String, colocateWith: Variable) =
//      non_slot_dict.compute(name) { _, v ->
//        v ?: tf.colocateWith(colocateWith) {
//          tf.variable(initial_value, name = name, trainable = false)
//        }
//      }
//
//  protected fun get_non_slot_variable(name: String): Variable {
//    val non_slot = non_slot_dict[name]
//    return non_slot!!
//  }
//}