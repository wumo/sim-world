import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.ops.TensorArray
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable

fun register_tensor_array_grad() {
  /**Gradients for operators defined in tensor_array_ops.py.*/
  registerNonDifferentiable("TensorArray")
  registerNonDifferentiable("TensorArrayGrad")
  registerNonDifferentiable("TensorArraySize")
  registerNonDifferentiable("TensorArrayClose")
  registerNonDifferentiable("TensorArrayV2")
  registerNonDifferentiable("TensorArrayGradV2")
  registerNonDifferentiable("TensorArraySizeV2")
  registerNonDifferentiable("TensorArrayCloseV2")
  registerNonDifferentiable("TensorArrayV3")
  registerNonDifferentiable("TensorArrayGradV3")
  registerNonDifferentiable("TensorArrayGradWithShape")
  registerNonDifferentiable("TensorArraySizeV3")
  registerNonDifferentiable("TensorArrayCloseV3")
  
  fun getGradSource(opOrTensorName: String): String {
    /**Identify which call to tf.gradients created this gradient op or tensor.
    
    TensorArray gradient calls use an accumulator TensorArray object.  If
    multiple gradients are calculated and run in the same session, the multiple
    gradient nodes may accidentally flow throuth the same accumulator TensorArray.
    This double counting breaks the TensorArray gradient flow.
    
    The solution is to identify which gradient call this particular
    TensorArray*Grad is being called in, by looking at the input gradient
    tensor's name, and create or lookup an accumulator gradient TensorArray
    associated with this specific call.  This solves any confusion and ensures
    different gradients from the same forward graph get their own accumulators.
    
    This function creates the unique label associated with the tf.gradients call
    that is used to create the gradient TensorArray.
    
    Args:
    op_or_tensor: `Tensor` or `Operation` which is an input to a
    TensorArray*Grad call.
    
    Returns:
    A python string, the unique label associated with this particular
    gradients calculation.
    
    Raises:
    ValueError: If not called within a gradients calculation.
     */
    val nameTokens = opOrTensorName.split("/")
    
    nameTokens.last { it.startsWith("gradients") }
    val gradPos = nameTokens.withIndex().lastOrNull { (_, x) ->
      x.startsWith("gradients")
    } ?: throw InvalidArgumentException(
        "Expected op/tensor name to start with 'Gradient' (excluding native), but got instead: $opOrTensorName.")
    return nameTokens.take(gradPos.index + 1).joinToString("/")
  }
  register("TensorArrayRead", "TensorArrayReadV2", "TensorArrayReadV3") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for TensorArrayRead.
    
    Args:
    op: Forward TensorArrayRead op.
    grad: Gradient `Tensor` to TensorArrayRead.
    
    Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
     */
    val handle = op.inputs[0]
    val index = op.inputs[1]
    val flow = op.inputs[2]
    val dtype = op.attrDataType("dtype")
    val gradSource = getGradSource(grad.name)
    val g = TensorArray.createFromHandle(handle, flow, dtype, colocateWithFirstWriteCall = false)
        .gradient(source = gradSource, flow = flow)
    val wG = g.write(index, grad)
    listOf(null, null, wG.flow) //return@register
  }
  register("TensorArrayWrite", "TensorArrayWriteV2", "TensorArrayWriteV3") { op, grad ->
    val flow = grad[0]!!.toOutput()
    /**Gradient for TensorArrayWrite.
    
    Args:
    op: Forward TensorArrayWrite op.
    flow: Gradient `Tensor` flow to TensorArrayWrite.
    
    Returns:
    A grad `Tensor`, the gradient created in an upstream ReadGrad or PackGrad.
     */
    val handle = op.inputs[0]
    val index = op.inputs[1]
    val dtype = op.attrDataType("T")
    val gradSource = getGradSource(flow.name)
    val g = TensorArray.createFromHandle(dataType = dtype, handle = handle, flow = flow,
                                         colocateWithFirstWriteCall = false)
        .gradient(source = gradSource, flow = flow)
    val grad = g.read(index)
    listOf(null, null, grad, flow) //return@register
  }
  register("TensorArrayGather", "TensorArrayGatherV2", "TensorArrayGatherV3") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for TensorArrayGather.
    
    Args:
    op: Forward TensorArrayGather op.
    grad: Gradient `Tensor` to TensorArrayGather.
    
    Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
     */
    val handle = op.inputs[0]
    val indices = op.inputs[1]
    val flow = op.inputs[2]
    val dtype = op.attrDataType("dtype")
    val gradSource = getGradSource(grad.name)
    val g = TensorArray.createFromHandle(dataType = dtype, handle = handle, flow = flow,
                                         colocateWithFirstWriteCall = false)
        .gradient(source = gradSource, flow = flow)
    val uG = g.scatter(indices, grad)
    listOf(null, null, uG.flow) //return@register
  }
  register("TensorArrayScatter", "TensorArrayScatterV2", "TensorArrayScatterV3") { op, grad ->
    val flow = grad[0]!!.toOutput()
    /**Gradient for TensorArrayScatter.
    
    Args:
    op: Forward TensorArrayScatter op.
    flow: Gradient `Tensor` flow to TensorArrayScatter.
    
    Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
     */
    val handle = op.inputs[0]
    val indices = op.inputs[1]
    val dtype = op.attrDataType("T")
    val gradSource = getGradSource(flow.name)
    val g = TensorArray.createFromHandle(dataType = dtype, handle = handle, flow = flow,
                                         colocateWithFirstWriteCall = false)
        .gradient(source = gradSource, flow = flow)
    val grad = g.gather(indices)
    listOf(null, null, grad, flow) //return@register
  }
  register("TensorArrayConcat", "TensorArrayConcatV2", "TensorArrayConcatV3") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for TensorArrayConcat.
    
    Args:
    op: Forward TensorArrayConcat op.
    grad: Gradient `Tensor` to TensorArrayConcat.
    
    Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
     */
    val handle = op.inputs[0]
    val flow = op.inputs[1]
    val lengths = op.outputs[1]
    val dtype = op.attrDataType("dtype")
    val gradSource = getGradSource(grad.name)
    val g = TensorArray.createFromHandle(dataType = dtype, handle = handle, flow = flow,
                                         colocateWithFirstWriteCall = false)
        .gradient(source = gradSource, flow = flow)
    val uG = g.split(grad, lengths = lengths)
    listOf(null, uG.flow) //return@register
  }
  register("TensorArraySplit", "TensorArraySplitV2", "TensorArraySplitV3") { op, grad ->
    val flow = grad[0]!!.toOutput()
    /**Gradient for TensorArraySplit.
    
    Args:
    op: Forward TensorArraySplit op.
    flow: Gradient `Tensor` flow to TensorArraySplit.
    
    Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
     */
    val handle = op.inputs[0]
    val dtype = op.attrDataType("T")
    val gradSource = getGradSource(flow.name)
    val g = TensorArray.createFromHandle(dataType = dtype, handle = handle, flow = flow,
                                         colocateWithFirstWriteCall = false)
        .gradient(source = gradSource, flow = flow)
    val grad = g.concat()
    listOf(null, grad, null, flow) //return@register
  }
}