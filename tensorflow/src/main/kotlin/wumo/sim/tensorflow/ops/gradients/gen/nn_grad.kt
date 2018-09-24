import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike
import wumo.sim.tensorflow.ops.basic.*
import wumo.sim.tensorflow.ops.gen.gen_nn_ops
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.tensor.constantValue
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.FLOAT16
import wumo.sim.tensorflow.types.INT32
import wumo.sim.tensorflow.types.INT64
import wumo.sim.util.i
import wumo.sim.util.scalarDimension

fun register_nn_grad() {
  /**Gradients for operators defined in nn_ops.py.*/
  register("Conv2DBackpropInput") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**The derivatives for deconvolution.
    
    Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output
    
    Returns:
    the gradients w.r.t. the input and the filter
     */
    listOf(null, tf.conv2DBackpropInput(grad,
                                        tf.shape(op.inputs[1]),
                                        op.inputs[2],
                                        dilations = op.attrLongList("dilations"),
                                        strides = op.attrLongList("strides"),
                                        padding = op.attrString("padding"),
                                        useCudnnOnGpu = op.attrBool("use_cudnn_on_gpu"),
                                        dataFormat = op.attrString("data_format")),
           tf.conv2D(grad,
                     op.inputs[1],
                     dilations = op.attrLongList("dilations"),
                     strides = op.attrLongList("strides"),
                     padding = op.attrString("padding"),
                     useCudnnOnGpu = op.attrBool("use_cudnn_on_gpu"),
                     dataFormat = op.attrString("data_format"))) //return@register
  }
  register("Conv2DBackpropFilter") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.conv2DBackpropInput(tf.shape(op.inputs[0]),
                                  grad,
                                  op.inputs[2],
                                  dilations = op.attrLongList("dilations"),
                                  strides = op.attrLongList("strides"),
                                  padding = op.attrString("padding"),
                                  useCudnnOnGpu = op.attrBool("use_cudnn_on_gpu"),
                                  dataFormat = op.attrString("data_format")),
           null, tf.conv2D(op.inputs[0],
                           grad,
                           dilations = op.attrLongList("dilations"),
                           strides = op.attrLongList("strides"),
                           padding = op.attrString("padding"),
                           useCudnnOnGpu = op.attrBool("use_cudnn_on_gpu"),
                           dataFormat = op.attrString("data_format"))) //return@register
  }
  register("Conv3D") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val dataFormat = op.attrString("data_format")
    listOf(tf.conv3DBackpropFilterV2(tf.shape(op.inputs[0]),
                                     op.inputs[1],
                                     grad,
                                     dilations = op.attrLongList("dilations"),
                                     strides = op.attrLongList("strides"),
                                     padding = op.attrString("padding"),
                                     dataFormat = dataFormat),
           tf.conv3DBackpropFilterV2(op.inputs[0],
                                     tf.shape(op.inputs[1]),
                                     grad,
                                     dilations = op.attrLongList("dilations"),
                                     strides = op.attrLongList("strides"),
                                     padding = op.attrString("padding"),
                                     dataFormat = dataFormat)) //return@register
  }
  register("Conv3DBackpropInputV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val dataFormat = op.attrString("data_format")
    listOf(null, tf.conv3DBackpropFilterV2(grad,
                                           tf.shape(op.inputs[1]),
                                           op.inputs[2],
                                           dilations = op.attrLongList("dilations"),
                                           strides = op.attrLongList("strides"),
                                           padding = op.attrString("padding"),
                                           dataFormat = dataFormat),
           tf.conv3D(grad, op.inputs[1],
                     dilations = op.attrLongList("dilations"),
                     strides = op.attrLongList("strides"),
                     padding = op.attrString("padding"),
                     dataFormat = dataFormat)) //return@register
  }
  register("Conv3DBackpropFilterV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val dataFormat = op.attrString("data_format")
    listOf(tf.conv3DBackpropInputV2(tf.shape(op.inputs[0]),
                                    grad,
                                    op.inputs[2],
                                    dilations = op.attrLongList("dilations"),
                                    strides = op.attrLongList("strides"),
                                    padding = op.attrString("padding"),
                                    dataFormat = dataFormat),
           null, tf.conv3D(op.inputs[0], grad,
                           dilations = op.attrLongList("dilations"),
                           strides = op.attrLongList("strides"),
                           padding = op.attrString("padding"),
                           dataFormat = dataFormat)) //return@register
  }
  register("AvgPool3D") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.avgPool3DGrad(tf.shape(op.inputs[0]),
                            grad,
                            ksize = op.attrLongList("ksize"),
                            strides = op.attrLongList("strides"),
                            padding = op.attrString("padding"),
                            dataFormat = op.attrString("data_format")))  //return@register
  }
  register("AvgPool3DGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.stopGradient(op.inputs[0]),
           tf.avgPool3D(grad,
                        op.attrLongList("ksize"),
                        op.attrLongList("strides"),
                        op.attrString("padding"),
                        dataFormat = op.attrString("data_format")))  //return@register
  }
  register("MaxPool3D") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.maxPool3DGrad(op.inputs[0], op.outputs[0], grad,
                            ksize = op.attrLongList("ksize"),
                            strides = op.attrLongList("strides"),
                            padding = op.attrString("padding"),
                            dataFormat = op.attrString("data_format")))  //return@register
  }
  register("MaxPool3DGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.zeros(shape = tf.shape(op.inputs[0]),
                    dtype = op.inputs[0].dataType),
           tf.zeros(shape = tf.shape(op.inputs[1]),
                    dtype = op.inputs[1].dataType),
           tf.maxPool3DGradGrad(op.inputs[0], op.inputs[1], grad,
                                op.attrLongList("ksize"),
                                op.attrLongList("strides"),
                                padding = op.attrString("padding"),
                                dataFormat = op.attrString("data_format")))  //return@register
  }
  register("MaxPool3DGradGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.zeros(shape = tf.shape(op.inputs[0]),
                    dtype = op.inputs[0].dataType),
           tf.zeros(shape = tf.shape(op.inputs[1]),
                    dtype = op.inputs[1].dataType),
           tf.maxPool3DGrad(op.inputs[0], op.inputs[1], grad,
                            op.attrLongList("ksize"),
                            op.attrLongList("strides"),
                            padding = op.attrString("padding"),
                            dataFormat = op.attrString("data_format")))  //return@register
  }
  register("Softmax") { op, grad ->
    val gradSoftmax = grad[0]!!.toOutput()
    /**The derivative of the softmax nonlinearity.
    
    We assume that probs is of shape [batch_size * dim]
    The formula for dsoftmax / dx = (diag(softmax) - softmax * softmax').
    This matrix is diagonal minus a rank one matrix, so it is easy to implement
    as follows:
    
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax
    
    Args:
    op: the Softmax op.
    grad_softmax:  the tensor representing the gradient w.r.t. the
    softmax output.
    
    Returns:
    gradient w.r.t the input to the softmax
     
     */
    val softmax = op.outputs[0]
    val sumChannels = tf.sum(gradSoftmax * softmax, tf.const(-1), keepDims = true)
    listOf((gradSoftmax - sumChannels) * softmax)  //return@register
  }
  register("LogSoftmax") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**The gradient for log_softmax.
    
    log_softmax = input - log(sum(exp(input))
    dlog_softmax/dinput = diag - softmax(input)
    
    Args:
    op: The log softmax op.
    grad: The tensor representing the gradient w.r.t. the output.
    
    Returns:
    The gradients w.r.t. the input.
     */
    val softmax = tf.exp(op.outputs[0])
    listOf(grad - tf.sum(grad, tf.const(-1), keepDims = true) * softmax)  //return@register
  }
  register("BiasAdd") { op, grad ->
    val receivedGrad = grad[0]!!.toOutput()
    /**Return the gradients for the 2 inputs of bias_op.
    
    The first input of unused_bias_op is the tensor t, and its gradient is
    just the gradient the unused_bias_op received.
    
    The second input of unused_bias_op is the bias vector which has one fewer
    dimension than "received_grad" (the batch dimension.)  Its gradient is the
    received gradient Summed on the batch dimension, which is the first dimension.
    
    Args:
    op: The BiasOp for which we need to generate gradients.
    received_grad: Tensor.  The gradients passed to the BiasOp.
    
    Returns:
    Two tensors, the first one for the "tensor" input of the BiasOp,
    the second one for the "bias" input of the BiasOp.
     */
    val dataFormat = try {
      op.attrString("data_format")
    } catch (_: Exception) {
      "NHWC"
    }
    listOf(receivedGrad, gen_nn_ops.biasAddGrad(
        outBackprop = receivedGrad, dataFormat = dataFormat))  //return@register
  }
  register("BiasAddGrad") { op, grad ->
    val receivedGrad = grad[0]!!.toOutput()
    /**Gradient for the BiasAddGrad op.
    
    Args:
    op: BiasAddGrad op for which we are calculating gradients.
    received_grad: The gradients passed to the BiasAddGrad op.
    
    Returns:
    A single gradient Tensor for the input to BiasAddGrad (which
    is the gradient of the bias term in BiasAdd)
     */
    
    val dataFormat = try {
      op.attrString("data_format")
    } catch (_: Exception) {
      "NCHW"
    }
    val shape = tf.shape(op.inputs[0])
    val rank = tf.rank(op.inputs[0])
    val biasShape = tf.shape(receivedGrad)
    val zero = tf.const(0)
    lateinit var expandedShape: Output
    lateinit var tileMults: Output
    if (dataFormat == "NCHW") {
      val valuesLeft = shape.slice(0, -3)
      val valuesRight = shape.slice(-2)
      expandedShape = tf.concat(listOf(
          tf.onesLike(valuesLeft), biasShape,
          tf.onesLike(valuesRight)), zero)
      tileMults = tf.concat(listOf(valuesLeft, tf.const(1), valuesRight), zero)
    } else {
      val valuesLeft = shape.slice(0, -1)
      expandedShape = tf.concat(listOf(tf.onesLike(valuesLeft), biasShape), zero)
      tileMults = tf.concat(listOf(valuesLeft, tf.const(1)), zero)
    }
    val expandedGrad = tf.reshape(receivedGrad, expandedShape)
    listOf(tf.tile(expandedGrad, tileMults))  //return@register
  }
  register("BiasAddV1") { op, grad ->
    val receivedGrad = grad[0]!!.toOutput()
    /**Return the gradients for the 2 inputs of bias_op.
    
    The first input of unused_bias_op is the tensor t, and its gradient is
    just the gradient the unused_bias_op received.
    
    The second input of unused_bias_op is the bias vector which has one fewer
    dimension than "received_grad" (the batch dimension.)  Its gradient is the
    received gradient Summed on the batch dimension, which is the first dimension.
    
    Args:
    unused_bias_op: The BiasOp for which we need to generate gradients.
    received_grad: Tensor.  The gradients passed to the BiasOp.
    
    Returns:
    Two tensors, the first one for the "tensor" input of the BiasOp,
    the second one for the "bias" input of the BiasOp.
     */
    val reductionDimTensor = tf.range(tf.const(0), tf.rank(receivedGrad) - 1)
    listOf(receivedGrad, tf.sum(receivedGrad, reductionDimTensor))  //return@register
  }
  register("Relu") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(gen_nn_ops.reluGrad(grad, op.outputs[0]))  //return@register
  }
  register("EluGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val eluX = op.inputs[1]
    listOf(tf.eluGrad(grad, op.outputs[0]),
           tf.where(tf.less(eluX, tf.const(0)),
                    grad * op.inputs[0],
                    tf.zeros(shape = tf.shape(eluX), dtype = eluX.dataType)))  //return@register
  }
  register("SeluGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[1]
    val scaleAlpha = 1.7580993408473768599402175208123
    listOf(tf.eluGrad(grad, op.outputs[0]),
           tf.where(tf.less(x, tf.const(0.0)),
                    tf.eluGrad(grad, op.outputs[0] + scaleAlpha),
                    tf.zeros(shape = tf.shape(x), dtype = x.dataType)))  //return@register
  }
  register("Relu6") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.relu6Grad(grad, op.outputs[0]))  //return@register
  }
  register("Relu6Grad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[1]
    listOf(tf.relu6Grad(grad, x), tf.zeros(shape = tf.shape(x), dtype = x.dataType))  //return@register
  }
  register("Elu") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.eluGrad(grad, op.outputs[0]))  //return@register
  }
  register("Selu") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.seluGrad(grad, op.outputs[0]))  //return@register
  }
  register("Softplus") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.softplusGrad(grad, op.inputs[0]))  //return@register
  }
  register("SoftplusGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val (dy, x) = op.inputs
    tf.controlDependencies(grad) {
      val ddy = tf.softplusGrad(grad, x)
      val d2x = grad * dy / (tf.exp(-x) + 2.0 + tf.exp(x))
      listOf(ddy, d2x)  //return@register
    }
  }
  register("Softsign") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.softsignGrad(grad, op.inputs[0]))  //return@register
  }
  register("ReluGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[1]
    listOf(tf.reluGrad(grad, x), tf.zeros(shape = tf.shape(x), dtype = x.dataType))  //return@register
  }
  fun broadcastMul(vec: Output, mat: Output): Output {
    /**Multiply after broadcasting vec to match dimensions of mat.
    
    Args:
    vec: A 1-D tensor of dimension [D0]
    mat: A 2-D tensor of dimension [D0, D1]
    
    Returns:
    A tensor of dimension [D0, D1], the result of vec * mat
     */
    return tf.expandDims(vec, tf.const(-1)) * mat
  }
  register("SoftmaxCrossEntropyWithLogits") { op, grad ->
    val gradLoss = grad[0]!!.toOutput()
    val gradGrad = grad[1]!!.toOutput()
    /**Gradient function for SoftmaxCrossEntropyWithLogits.*/
    val softmaxGrad = op.outputs[1]
    var grad = broadcastMul(gradLoss, softmaxGrad)
    fun isZero(g: Output) =
        if (g.op.opType == "ZerosLike" || g.op.opType == "Zeros") {
          true
        } else {
          val constFillValue = constantValue<Any>(g)
          constFillValue != null && constFillValue.all { it == 0 }
        }
    
    val logits = op.inputs[0]
    if (gradGrad != null && !isZero(gradGrad)) {
      val softmax = tf.softmax(logits)
      grad += ((gradGrad - tf.squeeze(tf.matMul(tf.expandDims(gradGrad, tf.const(1)),
                                                tf.expandDims(softmax, tf.const(2))), arrayOf(1L))) * softmax)
    }
    listOf(grad, broadcastMul(gradLoss, -tf.logSoftmax(logits)))  //return@register
  }
  register("SparseSoftmaxCrossEntropyWithLogits") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient function for SparseSoftmaxCrossEntropyWithLogits.*/
    val sparseSoftmaxGradWithoutGradient =
        tf.preventGradient(op.outputs[1],
                           message = "Currently there is no way to take the second " +
                               "derivative of sparse_softmax_cross_entropy_with_logits due to the fused " +
                               "implementation's interaction with tf.gradients()")
    listOf(broadcastMul(grad, sparseSoftmaxGradWithoutGradient), null)  //return@register
  }
  register("Conv2D") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val dilations = op.attrLongList("dilations")
    val strides = op.attrLongList("strides")
    val padding = op.attrString("padding")
    val useCudnnOnGpu = op.attrBool("use_cudnn_on_gpu")
    val dataFormat = op.attrString("data_format")
    val (shape0, shape1) = tf.shapeN(listOf(op.inputs[0], op.inputs[1]))
    listOf(tf.conv2DBackpropInput(shape0, op.inputs[1], grad,
                                  dilations = dilations,
                                  strides = strides,
                                  padding = padding,
                                  useCudnnOnGpu = useCudnnOnGpu,
                                  dataFormat = dataFormat),
           tf.conv2DBackpropFilter(op.inputs[0], shape1, grad,
                                   dilations = dilations,
                                   strides = strides,
                                   padding = padding,
                                   useCudnnOnGpu = useCudnnOnGpu,
                                   dataFormat = dataFormat)) //return@register
  }
  register("DepthwiseConv2dNative") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.depthwiseConv2dNativeBackpropInput(tf.shape(op.inputs[0]),
                                                 op.inputs[1],
                                                 grad,
                                                 op.attrLongList("strides"),
                                                 op.attrString("padding"),
                                                 dataFormat = op.attrString("data_format")),
           tf.depthwiseConv2dNativeBackpropFilter(op.inputs[0], tf.shape(op.inputs[1]), grad,
                                                  op.attrLongList("strides"),
                                                  op.attrString("padding"),
                                                  dataFormat = op.attrString("data_format"))) //return@register
  }
  register("Dilation2D") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.dilation2DBackpropInput(op.inputs[0],
                                      op.inputs[1],
                                      grad,
                                      op.attrLongList("strides"),
                                      op.attrLongList("rates"),
                                      op.attrString("padding")),
           tf.dilation2DBackpropFilter(op.inputs[0],
                                       op.inputs[1],
                                       grad,
                                       op.attrLongList("strides"),
                                       op.attrLongList("rates"),
                                       op.attrString("padding"))) //return@register
  }
  register("LRN") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val depthRadius = op.attrLong("depth_radius")
    val bias = op.attrFloat("bias")
    val alpha = op.attrFloat("alpha")
    val beta = op.attrFloat("beta")
    listOf(tf.lRNGrad(grad, op.inputs[0], op.outputs[0], depthRadius, bias, alpha, beta)) //return@register
  }
  register("AvgPool") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.avgPoolGrad(tf.shape(op.inputs[0]),
                          grad,
                          op.attrLongList("ksize"),
                          op.attrLongList("strides"),
                          op.attrString("padding"),
                          dataFormat = op.attrString("data_format")))  //return@register
  }
  register("AvgPoolGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.stopGradient(op.inputs[0]),
           tf.avgPool(grad,
                      op.attrLongList("ksize"),
                      op.attrLongList("strides"),
                      op.attrString("padding"),
                      dataFormat = op.attrString("data_format")))  //return@register
  }
  register("MaxPool") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.maxPoolGrad(op.inputs[0],
                          op.outputs[0], grad,
                          op.attrLongList("ksize"),
                          op.attrLongList("strides"),
                          padding = op.attrString("padding"),
                          dataFormat = op.attrString("data_format")))  //return@register
  }
  register("MaxPoolV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val ksize = op.inputs[1]
    val strides = op.inputs[2]
    listOf(tf.maxPoolGradV2(op.inputs[0], op.outputs[0], grad, ksize, strides,
                            padding = op.attrString("padding"),
                            dataFormat = op.attrString("data_format")), null, null)  //return@register
  }
  register("MaxPoolWithArgmax") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.maxPoolGradWithArgmax(op.inputs[0], grad, op.outputs[1],
                                    op.attrLongList("ksize"),
                                    op.attrLongList("strides"),
                                    padding = op.attrString("padding")))  //return@register
  }
  register("MaxPoolGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.zeros(shape = tf.shape(op.inputs[0]), dtype = op.inputs[0].dataType),
           tf.zeros(shape = tf.shape(op.inputs[1]), dtype = op.inputs[1].dataType),
           tf.maxPoolGradGrad(op.inputs[0], op.inputs[1], grad,
                              op.attrLongList("ksize"),
                              op.attrLongList("strides"),
                              padding = op.attrString("padding"),
                              dataFormat = op.attrString("data_format")))  //return@register
  }
  register("MaxPoolGradV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val ksize = op.inputs[3]
    val strides = op.inputs[4]
    listOf(tf.zeros(shape = tf.shape(op.inputs[0]), dtype = op.inputs[0].dataType),
           tf.zeros(shape = tf.shape(op.inputs[1]), dtype = op.inputs[1].dataType),
           tf.maxPoolGradGradV2(op.inputs[0], op.inputs[1], grad, ksize, strides,
                                padding = op.attrString("padding"),
                                dataFormat = op.attrString("data_format")), null, null)  //return@register
  }
  register("MaxPoolGradGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.zeros(shape = tf.shape(op.inputs[0]), dtype = op.inputs[0].dataType),
           tf.zeros(shape = tf.shape(op.inputs[1]), dtype = op.inputs[1].dataType),
           tf.maxPoolGrad(op.inputs[0], op.inputs[1], grad,
                          op.attrLongList("ksize"),
                          op.attrLongList("strides"),
                          padding = op.attrString("padding"),
                          dataFormat = op.attrString("data_format")))  //return@register
  }
  register("FractionalMaxPool") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Returns gradient for FractionalMaxPool.
    
    Since FractionalMaxPool has three outputs, there are three gradients passed in
    for each of the outputs. Only the first one is useful, the other two gradients
    are empty.
    
    Args:
    op: The FractionalMaxPoolOp.
    grad_0: Gradient with respect to op.outputs[0]
    unused_grad_1: Gradient with respect to op.outputs[1]/row_seq. It is empty.
    unused_grad_2: Gradient with respect to op.outputs[2]/col_seq. It is empty.
    
    Returns:
    Input backprop for FractionalMaxPool op.
     */
    listOf(tf.fractionalMaxPoolGrad(op.inputs[0], op.outputs[0], grad, op.outputs[1], op.outputs[2],
                                    op.attrBool("overlapping")))  //return@register
  }
  register("FractionalAvgPool") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Returns gradient for FractionalAvgPool.
    
    Since FractionalAvgPool has three outputs, there are three gradients passed in
    for each of the outputs. Only the first one is useful, the other two gradients
    are empty.
    
    Args:
    op: The FractionalAvgPoolOp.
    grad_0: Gradient with respect to op.outputs[0]
    unused_grad_1: Gradient with respect to op.outputs[1]/row_seq. It is empty.
    unused_grad_2: Gradient with respect to op.outputs[2]/col_seq. It is empty.
    
    Returns:
    Input backprop for FractionalAvgPool op.
     */
    listOf(tf.fractionalAvgPoolGrad(op.inputs[0].shape.toOutput(), grad, op.outputs[1], op.outputs[2],
                                    op.attrBool("overlapping")))  //return@register
  }
  register("BatchNormWithGlobalNormalization") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Return the gradients for the 5 inputs of BatchNormWithGlobalNormalization.
    
    We do not backprop anything for the mean and var intentionally as they are
    not being trained with backprop in the operation.
    
    Args:
    op: The BatchNormOp for which we need to generate gradients.
    grad: Tensor.  The gradients passed to the BatchNormOp.
    
    Returns:
    dx: Backprop for input, which is (grad * (g * rsqrt(v + epsilon)))
    dm: Backprop for mean, which is
    sum_over_rest(grad * g) * (-1 / rsqrt(v + epsilon))
    dv: Backprop for variance, which is
    sum_over_rest(grad * g * (x - m)) * (-1/2) * (v + epsilon) ^ (-3/2)
    db: Backprop for beta, which is grad reduced in all except the
    last dimension.
    dg: Backprop for gamma, which is (grad * ((x - m) * rsqrt(v + epsilon)))
     */
    val (dx, dm, dv, db, dg) = tf.batchNormWithGlobalNormalizationGrad(op.inputs[0],
                                                                       op.inputs[1],
                                                                       op.inputs[2],
                                                                       op.inputs[4],
                                                                       grad,
                                                                       op.attrFloat("variance_epsilon"),
                                                                       op.attrBool("scale_after_normalization"))
    listOf(dx, dm, dv, db, dg)  //return@register
  }
  fun baseFusedBatchNormGrad(op: Op, useV2: Boolean, grad: List<OutputLike?>): List<OutputLike?> {
    /**Return the gradients for the 3 inputs of BatchNorm.
    
    Args:
    op: The BatchNormOp for which we need to compute gradients.
    use_v2: Boolean indicating whether to use the V2 version of the fused batch
    norm gradient.
     *grad: An argument list for tensors of gradients wrt the outputs
    with grad[0] as grad_y.
    
    Returns:
    grad_x: gradient for x, which is scale * rsqrt(variance + epsilon) *
    [grad_y - mean(grad_y) - (x - mean(x)) *
    mean(grad_y * (x - mean(x))) / (variance + epsilon)]
    in training mode; grad_y * scale * rsqrt(pop_variance + epsilon)
    in freeze mode.
    
    grad_scale: gradient for scale, which is sum(grad_y * (x - mean(x)) *
    rsqrt(variance + epsilon)) in training mode;
    sum(grad_y * (x - pop_mean) * rsqrt(pop_variance + epsilon))
    in freeze mode.
    
    grad_offset: gradient for offset, which is sum(grad_y) in training mode;
    sum(grad_y) in freeze mode.
     */
    var x = op.inputs[0]
    var gradY = grad[0]!!.toOutput()
    val scale = op.inputs[1]
    val epsilon = op.attrFloat("epsilon")
    val dataFormat = op.attrString("data_format")
    val isTraining = op.attrBool("is_training")
    val gradFun = if (useV2) tf::fusedBatchNormGradV2 else tf::fusedBatchNormGrad
    return if (isTraining) {
      gradFun(gradY, x, scale, op.outputs[3], op.outputs[4],
              epsilon,
              dataFormat,
              isTraining, "FusedBatchNormGrad")
    } else {
      val popMean = op.inputs[3]
      val popVar = op.inputs[4]
      if (dataFormat == "NCHW") {
        x = tf.transpose(x, tf.const(i(0, 2, 3, 1)))
        gradY = tf.transpose(gradY, tf.const(i(0, 2, 3, 1)))
      }
      var (dx, dscale, doffset, _, _) = gradFun(gradY, x, scale, popMean, popVar,
                                                epsilon, "NHWC", isTraining, "FusedBatchNormGrad")
      if (dataFormat == "NCHW")
        dx = tf.transpose(dx, tf.const(i(0, 3, 1, 2)))
      listOf(dx, dscale, doffset, null, null)
    }
  }
  register("FusedBatchNorm") { op, grad ->
    baseFusedBatchNormGrad(op, false, grad)  //return@register
  }
  register("FusedBatchNormV2") { op, grad ->
    baseFusedBatchNormGrad(op, true, grad)  //return@register
  }
  fun batchNormGrad(gradY: Output, x: Output, scale: Output,
                    popMean: Output, popVar: Output,
                    epsilon: Float, dataFormat: String, isTraining: Boolean = true): List<Output> {
    /**Returns the gradients for the 3 inputs of BatchNorm.
    
    Args:
    grad_y: A `Tensor` of 4 dimensions for gradient for y.
    x: A `Tensor` of 4 dimensions for x.
    scale: A `Tensor` of 1 dimension for scaling.
    pop_mean: A `Tensor` of 1 dimension for the population mean. Only used when
    is_training=False.
    pop_var: A `Tensor` of 1 dimension for the population variance. Only used
    when is_training=False.
    epsilon: A small float number added to the variance of x.
    data_format: The data format for input. Either b"NHWC" or b"NCHW".
    is_training: A bool value to indicate the operation is for training
    (default)
    or inference.
    
    Returns:
    A tuple (grad_x, grad_scale, grad_offset), where grad_x is the gradient
    for x, grad_scale the gradient for scale, and grad_offset the gradient
    for offset.
     */
    val xDtype = x.dataType.baseDataType
    var x = x
    var gradY = gradY
    var scale = scale
    var popMean = popMean
    var popVar = popVar
    if (xDtype == FLOAT16) {
      x = tf.cast(x, FLOAT)
      gradY = tf.cast(gradY, FLOAT)
    }
    lateinit var reduceAxis: LongArray
    var keepdims: Boolean = false
    return if (isTraining) {
      if (dataFormat == "NHWC") {
        keepdims = false
        reduceAxis = longArrayOf(0, 1, 2)
      } else {
        keepdims = true
        reduceAxis = longArrayOf(0, 2, 3)
        val shape = tf.stack(listOf(1, tf.size(scale), 1, 1))
        scale = tf.reshape(scale, shape)
      }
      val reduceAxisTensor=tf.const(reduceAxis)
      val meanGradY = tf.mean(gradY, reduceAxisTensor, keepDims = keepdims)
      val meanX = tf.mean(x, reduceAxisTensor, keepDims = keepdims)
      val varX = tf.mean(tf.squaredDifference(x, tf.stopGradient(meanX)), reduceAxisTensor, keepDims = keepdims)
      val gradYOffset = gradY - meanGradY
      val xOffset = x - meanX
      val mean = tf.mean(gradY * xOffset, axis = reduceAxisTensor, keepDims = keepdims)
      val gradX = scale * tf.rsqrt(varX + epsilon) * (gradYOffset - tf.reciprocal(varX + epsilon) * mean * xOffset)
      var gradScale = tf.rsqrt(varX + epsilon) * tf.sum(gradY * xOffset,
                                                        axis = tf.const(reduceAxis),
                                                        keepDims = keepdims)
      if (dataFormat == "NCHW")
        gradScale = tf.squeeze(gradScale)
      val gradOffset = tf.sum(gradY, axis = tf.const(reduceAxis))
      listOf(tf.cast(gradX, xDtype), gradScale, gradOffset)
    } else {
      if (dataFormat == "NHWC")
        reduceAxis = longArrayOf(0, 1, 2)
      else {
        reduceAxis = longArrayOf(0, 2, 3)
        val shape = tf.stack(listOf(1, tf.size(popMean), 1, 1))
        popMean = tf.reshape(popMean, shape)
        popVar = tf.reshape(popVar, shape)
        scale = tf.reshape(scale, shape)
      }
      val axis = tf.const(reduceAxis)
      val gradOffset = tf.sum(gradY, axis = axis)
      val varRsqrt = tf.rsqrt(popVar + epsilon)
      val gradScale = tf.sum(gradY * (x - popMean) * varRsqrt, axis = axis)
      val gradX = gradY * scale * varRsqrt
      listOf(tf.cast(gradX, xDtype), gradScale, gradOffset)
    }
  }
  register("FusedBatchNormGrad", "FusedBatchNormGradV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Returns the gradients for the 3 inputs of FusedBatchNormGrad.
    
    Args:
    op: The FusedBatchNormGradOp for which we need to compute gradients.
     *grad: An argument list for tensors of gradients wrt the outputs
    with grad[0] as grad_grad_x, grad[1] as grad_grad_scale,
    grad[2] as grad_grad_offset.
    
    Returns:
    A tuple (grad_grad_y, grad_x, grad_scale, None, None), where grad_grad_y
    is the gradient for grad_y, grad_x the gradient for x, grad_scale the
    gradient for scale.
     */
    val dataFormat = op.attrString("data_format")
    val epsilon = op.attrFloat("epsilon")
    val isTraining = op.attrBool("is_training")
    val gradY = op.inputs[0]
    val x = op.inputs[1]
    val scale = op.inputs[2]
    val popMean = op.inputs[3]
    val popVar = op.inputs[4]
    val gradGradX = grad[0]
    val gradGradScale = grad[1]
    val gradGradOffset = grad[2]
    val (_gradX, _gradScale, gradOffset) = batchNormGrad(gradY, x, scale, popMean, popVar, epsilon, dataFormat, isTraining)
    val gradInitial = listOf(gradGradX, gradGradScale, gradGradOffset)
    val (gradGradY, gradX, gradScale) = tf.gradients(listOf(_gradX, _gradScale, gradOffset),
                                                     listOf(gradY, x, scale), gradInitial)
    listOf(gradGradY, gradX, gradScale, null, null)  //return@register
  }
  register("L2Loss") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Return the gradients for L2Loss.
    
    Args:
    op: The L2LossOp for which we need to generate gradients.
    grad: Tensor containing a single number.
    
    Returns:
    The gradient, which is (x * grad).
     */
    listOf(op.inputs[0] * grad)  //return@register
  }
  register("TopK", "TopKV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Return the gradients for TopK.
    
    Args:
    op: The TopKOp for which we need to generate gradients.
    grad: Tensor. The gradients passed to the TopKOp.
    
    Returns:
    A list of two tensors, the first being the gradient w.r.t to the input and
    TopK, and the second being the gradient w.r.t. to the indices (all zero).
     */
    val inShape = tf.shape(op.inputs[0])
    val indShape = tf.shape(op.outputs[1])
    val indLastdim = tf.gather(tf.cast(indShape, INT64), tf.size(indShape) - 1)
    val ind2d = tf.reshape(op.outputs[1], tf.stack(listOf(-1, indLastdim)))
    val inLastdim = tf.gather(tf.cast(inShape, INT64), tf.size(inShape) - 1)
    val outerdim = tf.shape(ind2d)[0]
    val minusOne = tf.const(-1)
    val ind = tf.reshape(ind2d + tf.cast(
        tf.expandDims(tf.range(tf.const(0), tf.cast(outerdim, INT64) * inLastdim, inLastdim),
                      minusOne), INT32), minusOne)
    listOf(tf.reshape(tf.scatterNd(tf.expandDims(ind, minusOne), tf.reshape(grad, minusOne),
                                   tf.prod(inShape)), inShape), tf.zeros(scalarDimension, dtype = INT32)) //return@register
  }
  register("NthElement") { op, grad ->
    var grad = grad[0]!!.toOutput()
    /**Return the gradients for NthElement.
    
    Args:
    op: The NthElementOp for which we need to generate gradients.
    grad: Tensor. The gradients passed to the NthElementOp
    
    Returns:
    A list of two tensors, the first being the gradient w.r.t. the input,
    the second being the gradient w.r.t. the N (None).
     */
    val input = op.inputs[0]
    val output = op.outputs[0]
    val minusOne = tf.const(-1)
    val indicators = tf.cast(tf.equal(tf.expandDims(output, minusOne), input), grad.dataType)
    grad = tf.expandDims(grad, minusOne)
    val numSelected = tf.expandDims(tf.sum(indicators, minusOne), minusOne)
    listOf(tf.div(indicators, numSelected) * grad, null) //return@register
  }
}