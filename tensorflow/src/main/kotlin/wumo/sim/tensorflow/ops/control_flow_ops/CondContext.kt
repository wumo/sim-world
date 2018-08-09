package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.ops.IndexedSlices
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.SparseOutput
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops.isLoopExit
import wumo.sim.tensorflow.tf

class CondContext(val pred: Output,
                  val pivot: Output,
                  val branch: Int) : ControlFlowContext() {
  override val gradState: GradientLoopState?
    get() = TODO("not implemented")
  
  init {
    //Values considered to have been already seen in this context. pred is not
    //included in this context.
    values += pred.name
    external_values[pred.name] = pred
    values += pivot.name
    pivot.op!!.set_control_flow_context(this)
  }
  
  override fun addOp(op: Op) {
    if (op.inputs.isEmpty()) {
      _removeExternalControlEdges(op)
      op.addControlInput(pivot.op!!)
    } else {
      for (i in 0 until op.inputs.size) {
        val x = op.inputs[i]
        val real_x = addValue(x)
        if (real_x != x)
          op.updateInput(i, real_x)
      }
      _removeExternalControlEdges(op)
      if (op.graph.is_function(op.opType) || op.opType == "SymbolicGradient")
        op.addControlInput(pivot.op!!)
    }
    //Mark op's outputs as seen by this context and any outer contexts.
    val output_names = op.outputs.map { it.name }
    var ctxt: ControlFlowContext? = this
    while (ctxt != null) {
      ctxt.values.addAll(output_names)
      ctxt = ctxt.outerContext
    }
    if (outerContext != null || !isLoopExit(op))
      op.graph.prevent_fetching(op)
  }
  
  private fun _removeExternalControlEdges(op: Op) {
    //TODO Remove any external control dependency on this op
  }
  
  /**Add `val` to the current context and its outer context recursively.*/
  private fun addValue(v: Output): Output {
    return if (v.name in values) {
      //Use the real value if it comes from outer context. This is needed in
      //particular for nested conds.
      external_values[v.name] ?: v
    } else {
      var result = v
      values += v.name
      if (outerContext != null) {
        result = (outerContext as CondContext).addValue(v)
        values += result.name
        external_values[result.name] = result
      }
      tf.control_dependencies {
        result = tf._switchRefOrTensor(result, pred)[branch]
      }
      result.op!!.graph.prevent_fetching(result.op!!)
      
      values += result.name
      external_values[v.name] = result
      result
    }
  }
  
  fun buildCondTensor(v: Any): Output {
    return when (v) {
      is Op -> {//Use pivot as the proxy for this op.
        with_dependencies(v, output_tensor = pivot)
      }
      is IndexedSlices, is SparseOutput -> {
        TODO()
      }
      else -> processOutputTensor((v as Output).value())
    }
  }
  
  /**Add the subgraph defined by fn() to the graph.*/
  fun buildCondBranch(fn: () -> Any): Output {
    val original_result = fn()
    val result = buildCondTensor(original_result)
    return result
  }
  
  private fun processOutputTensor(v: Output): Output {
    var real_v = v
    if (v.name !in values) {
      values += v.name
      real_v = tf._switchRefOrTensor(v, pred)[branch]
      external_values[v.name] = real_v
    } else {
      val external_v = external_values[v.name]
      if (external_v != null)
        real_v = external_v
    }
    return real_v
  }
}