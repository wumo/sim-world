package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.tf

class WhileContext : ControlFlowContext() {
  override fun addValue(output: Output): Output {
    TODO("not implemented")
  }
  
  override val backPropagate: Boolean
    get() = TODO("not implemented")
  override val gradState: GradientLoopState?
    get() = TODO("not implemented")
  override val name: String
    get() = TODO("not implemented")
  
  
  override fun addOp(op: Op) {
    TODO("not implemented")
  }
  
  /**
   * We move any external control dependencies of the op to the loop pivot, to
   * ensure they get executed.
   * @see "tensorflow.python.ops.control_flow_ops.WhileContext#_AddOpInternal"
   */
  override fun addInternal(op: Op) {
    val externalInputs = if (op.numInputs == 0) {
      // Remove any external control dependencies on this op.
      val (controlInputs, externalInputs) = removeExternalControlEdges(op)
      // Add a control edge from the control pivot to this op.
      if (controlInputs.isEmpty())
        controlPivot?.let { op.addControlInput(it) }
      op.outputs.forEach { values += it.name }
      externalInputs
    } else {
      op.inputs.withIndex().forEach { (index, input) ->
        val realInput = addValue(input)
        if (realInput != input)
          op.updateInput(index, realInput)
      }
      // Remove any external control dependencies on this op.
      val (_, externalInputs) = removeExternalControlEdges(op)
      // Add a control dependency to the op if it only depends on loop invariants. That is to prevent loop invariants
      // from enabling ops that should not be executed.
      maybeAddControlDependency(op)
      op.outputs.forEach { values += it.name }
      externalInputs
    }
    // TODO: [CONTROL_FLOW] Stop ignoring ops with no outputs.
    // Use an identity to pull control inputs as data inputs. Note that we ignore ops which do not have any outputs.
    tf.controlDependencies {
      enter()
      val externalInputs = externalInputs.map { op ->
        tf._identity(op.outputs[0]).op!!
      }
      exit()
      externalInputs
    }.let { op.addControlInputs(it) }
    
    if (outerContext != null || !control_flow_ops.isLoopExit(op)) {
      op.graph.preventFetching(op)
      op.outputs.forEach { op.graph.preventFeeding(it) }
    }
  }
  
  /**
   * Add a control input to the op if it only depends on loop invariants.
   */
  private fun maybeAddControlDependency(op: Op) {
    //Determines if `op` needs a control dependency.
    if (op.controlInputs.isEmpty() &&
        ((op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient")
            || op.inputs.all { control_flow_ops.isLoopConstantEnter(it.op!!) }))
      controlPivot?.let { op.addControlInput(it) }
  }
}