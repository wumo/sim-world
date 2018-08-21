package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops.isLoopExit
import wumo.sim.tensorflow.tf
import wumo.sim.util.emptyMutableSet

class CondContext(val predicate: Output,
                  val pivot: Output,
                  val branch: Int,
                  name: String = "CondContext") : ControlFlowContext() {
  
  init {
    //Values considered to have been already seen in this context. predicate is not
    //included in this context.
    values += predicate.name
    external_values[predicate.name] = predicate
    values += pivot.name
    pivot.op!!.controlFlowContext = this
  }
  
  override val name = tf.currentGraph.uniqueName(name)
  
  override val controlPivot = pivot.op
  
  override val condContext = this
  
  override fun addInternal(op: Op) {
    if (op.numInputs == 0) {
      // Remove any external control dependencies on this op.
      removeExternalControlEdges(op)
      controlPivot?.let { op.addControlInput(it) }
    } else {
      op.inputs.withIndex().forEach { (index, input) ->
        val realInput = addValue(input)
        if (realInput != input)
          op.updateInput(index, realInput)
      }
      // Remove any external control dependencies on this op.
      removeExternalControlEdges(op)
      if (op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient")
        controlPivot?.let { op.addControlInput(it) }
    }
    //Mark op's outputs as seen by this context and any outer contexts.
    val output_names = op.outputs.map { it.name }
    var ctxt: ControlFlowContext? = this
    while (ctxt != null) {
      ctxt.values.addAll(output_names)
      ctxt = ctxt.outerContext
    }
    if (outerContext != null || !isLoopExit(op))
      op.graph.preventFetching(op)
  }
  
  override val backPropagate = whileContext()?.backPropagate == true
  
  override val gradState = whileContext()?.gradState
  
  /**Add `val` to the current context and its outer context recursively.*/
  override fun addValue(output: Output): Output {
    return if (output.name in values) {
      //Use the real value if it comes from outer context. This is needed in
      //particular for nested conds.
      external_values.getOrDefault(output.name, output)
    } else {
      values += output.name
      val switchInput = outerContext?.let {
        val result = it.addValue(output)
        values += result.name
        external_values[result.name] = result
        result
      } ?: output
      
      val result = tf.controlDependencies(emptyMutableSet()) {
        control_flow_ops._switchRefOrTensor(switchInput, predicate)[branch]
      }
      result.op!!.graph.preventFetching(result.op)
      result.op.controlFlowContext = this
      values += result.name
      external_values[output.name] = result
      result
    }
  }
  
  /**Add the subgraph defined by fn() to the graph.*/
  fun <T> buildCondBranch(fn: () -> T): Pair<T, List<OutputLike>> {
    val originalResult = fn()
        ?: throw IllegalArgumentException("The provide cond branch functions must have return values other than 'null'.")
    val res = CondContext.processOutput(originalResult, this)
    val res_flat = CondContext.flatten(res)
    return originalResult to res_flat
  }
  
  fun <T> buildCondTensor(v: T): Output {
    return when (v) {
      is Op -> processOp(v)
      is IndexedSlices, is SparseOutput -> TODO()
      is OutputLike -> processOutput(v.toOutput())
      else -> TODO()
    }
  }
  
  private fun processOp(op: Op): Output {
    //Use pivot as the proxy for this op.
    return tf.withDependencies(mutableSetOf(op), pivot)
  }
  
  /** Processes an op output used in a conditional branch. */
  private fun processOutput(output: Output): Output {
    return if (output.name !in values) {
      // Handle the special case of () -> x.
      values += output.name
      val switchInput = outerContext?.let {
        val result = it.addValue(output)
        values += result.name
        external_values[result.name] = result
        result
      } ?: output
      val realValue = control_flow_ops._switchRefOrTensor(switchInput, predicate)[branch]
      external_values[output.name] = realValue
      realValue
    } else
      external_values[output.name] ?: output
  }
  
  interface CollectionKey : Graph.Graph.Key<CondContext>
  
  companion object {
    object COND_CONTEXT : CollectionKey {
      override val name: String = "cond_context"
    }
    
    fun <T> processOutput(output: T, context: CondContext): Any =
        when (output) {
          is Op -> context.processOp(output)
          is IndexedSlices -> TODO()
          is SparseOutput -> TODO()
          is OutputLike -> context.processOutput(output.toOutput())
          else -> TODO()
        }
    
    fun <R> flatten(processedOutput: R): List<OutputLike> =
        when (processedOutput) {
          is Output -> listOf(processedOutput)
          is IndexedSlices -> TODO()
          is SparseOutput -> TODO()
          else -> TODO()
        }
    
    fun <T> unflatten(output: T, values: List<OutputLike>): T =
        when (output) {
          is Op -> values.first().op as T
          is Output, is IndexedSlices, is SparseOutput -> values.first() as T
          else -> TODO()
        }
  }
}