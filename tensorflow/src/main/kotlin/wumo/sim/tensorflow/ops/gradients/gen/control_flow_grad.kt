import wumo.sim.tensorflow.core.UnimplementedException
import wumo.sim.tensorflow.ops.IndexedSlices
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.SparseOutput
import wumo.sim.tensorflow.ops.control_flow_ops.CondContext
import wumo.sim.tensorflow.ops.control_flow_ops.WhileContext
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.RESOURCE

fun register_control_flow_grad() {
  /**Gradients for operators defined in control_flow_ops.py.*/
  register("Switch", "RefSwitch") { op, grad ->
    /**Gradients for a Switch op is calculated using a Merge op.
    
    If the switch is a loop switch, it will be visited twice. We create
    the merge on the first visit, and update the other input of the merge
    on the second visit. A next_iteration is also added on second visit.
     */
    val gradCtxt = tf.currentControlFlowContext
    val opCtxt = op.controlFlowContext
    when (opCtxt) {
      is WhileContext -> {
        var mergeGrad = gradCtxt?.gradLoopState?.switchMap?.get(op)
        when {
          mergeGrad != null -> {
            if (grad[1] != null)
              WhileContext.addNextIterationAndBackEdge(mergeGrad, grad[1]!!, enforceShapeInvariant = false)
            listOf(null, null)
          }
          grad[0] != null -> {
            mergeGrad = tf.merge(listOf(grad[0]!!, grad[0]!!), name = "b_switch")[0]
            gradCtxt?.gradLoopState?.switchMap?.put(op, mergeGrad)
            listOf(mergeGrad, null)
          }
          else -> listOf(null, null)
        }
      }
      is CondContext -> {
        val zeroGrad = grad[1 - opCtxt.branch]
        if (zeroGrad == null) {
          if (op.inputs[0].dataType == RESOURCE)
            listOf(tf.merge(List(2) { grad[opCtxt.branch]!! }, name = "cond_resource_grad")[0], null)
          else
            listOf(null, null)
        } else
          listOf(tf.merge(grad.requireNoNulls(), name = "cond_grad")[0], null)
      }
      else -> {
        val falseGrad = tf.switch(grad[0]!!, op.inputs[1])[0]
        val trueGrad = tf.switch(grad[1]!!, op.inputs[1])[1]
        listOf(tf.merge(listOf(falseGrad, trueGrad))[0], null)
      }
    }
  }
  register("Merge", "RefMerge") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradients for a Merge op are calculated using a Switch op.*/
    val inputOp = op.inputs[0].op
    val opCtxt = control_flow_ops.getOutputContext(inputOp)
    val gradCtxt = tf.currentControlFlowContext
    when (opCtxt) {
      is WhileContext ->
        control_flow_ops._switchRefOrTensor(grad, (gradCtxt as WhileContext).pivot!!)  //return@register
      is CondContext -> {
        var pred = opCtxt.predicate
        gradCtxt?.gradLoopState?.let {
          pred = it.historyMap.getOrPut(pred.name) {
            it.backwardContext.exit()
            val historyPred = it.addForwardAccumulator(pred)
            it.backwardContext.enter()
            val realPred = it.addBackwardAccumulatedValue(historyPred, pred)
            realPred
          }
        }
        control_flow_ops._switchRefOrTensor(grad, pred, name = "cond_grad")  //return@register
      }
      else ->
        (0 until op.numInputs).map {
          control_flow_ops._switchRefOrTensor(grad, tf.equal(op.outputs[1], tf.const(it)))[1]
        }
    }
  }
  register("Exit", "RefExit") { op, grad ->
    val grad = grad[0]!!
    /**Gradients for an exit op are calculated using an Enter op.*/
    val opCtxt = op.controlFlowContext
    val gradCtxt = tf.currentControlFlowContext!!
    when {
      !gradCtxt.backPropagate -> listOf(null)
      opCtxt?.gradLoopState != null -> throw UnimplementedException("Second-order gradients are not supported for while loops.")
      else -> {
        when (grad) {
          is Output -> gradCtxt.values += grad.name
          is IndexedSlices -> {
            gradCtxt.values += grad.indices.name
            gradCtxt.values += grad.values.name
            grad.denseShape?.let {
              gradCtxt.values += it.name
            }
          }
          is SparseOutput -> {
            gradCtxt.values += grad.indices.name
            gradCtxt.values += grad.values.name
            grad.denseShape?.let {
              gradCtxt.values += it.name
            }
          }
        }
        gradCtxt as WhileContext
        gradCtxt.enter()
        val result = tf.enter(grad, gradCtxt.name, isContant = false, parallelIterations = gradCtxt.parallelIterations,
                              name = "b_exit")
        when (result) {
          is Output -> gradCtxt.loopEnters += result
          is IndexedSlices -> {
            gradCtxt.loopEnters += result.indices
            gradCtxt.loopEnters += result.values
            result.denseShape?.let { gradCtxt.loopEnters += result.denseShape }
          }
          is SparseOutput -> {
            gradCtxt.loopEnters += result.indices
            gradCtxt.loopEnters += result.values
            result.denseShape?.let { gradCtxt.loopEnters += result.denseShape }
          }
        }
        gradCtxt.exit()
        listOf(result)
      }
    }
  }
  register("NextIteration", "RefNextIteration") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**A forward next_iteration is translated into a backprop identity.
    
    Note that the backprop next_iteration is added in switch grad.
     */
    listOf(grad)  //return@register
  }
  register("Enter", "RefEnter") { op, grad ->
    val grad = grad[0]!!
    /**Gradients for an Enter are calculated using an Exit op.
    
    For loop variables, grad is the gradient so just add an exit.
    For loop invariants, we need to add an accumulator loop.
     */
    val gradCtxt = tf.currentControlFlowContext
    gradCtxt!!
    when {
      !gradCtxt.backPropagate -> listOf(grad)
      gradCtxt.gradLoopState != null -> listOf(grad)
      op.attrBool("is_constant") ->
        listOf((gradCtxt as WhileContext).addBackwardAccumulator(op, grad))
      else -> {
        gradCtxt as WhileContext
        val result = tf.exit(grad)
        when (result) {
          is Output -> gradCtxt.loopEnters += result
          is IndexedSlices -> {
            gradCtxt.loopEnters += result.indices
            gradCtxt.loopEnters += result.values
            result.denseShape?.let { gradCtxt.loopEnters += result.denseShape }
          }
          is SparseOutput -> {
            gradCtxt.loopEnters += result.indices
            gradCtxt.loopEnters += result.values
            result.denseShape?.let { gradCtxt.loopEnters += result.denseShape }
          }
        }
        listOf(result).apply { gradCtxt.exitResult(this) }
      }
    }
  }
  register("LoopCond") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Stop backprop for the predicate of a while loop.*/
    listOf(null)  //return@register
  }
}