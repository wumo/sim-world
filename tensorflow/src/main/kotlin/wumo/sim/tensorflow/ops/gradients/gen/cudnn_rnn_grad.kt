import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.tf

fun register_cudnn_rnn_grad() {
  /**Gradients for CuudnnRNN operators.*/
/* from__future__importabsolute_import */
/* from__future__importdivision */
/* from__future__importprint_function */
/* fromtensorflow.python.frameworkimportops */
/* fromtensorflow.python.opsimportgen_cudnn_rnn_ops */
  register("CudnnRNN") { op, grad ->
    val grads = grad.map { it!!.toOutput() }
    /**Gradients for the CudnnRNN op.*/
    if (!op.attrBool("is_training"))
      error("To use CudnnRNN in gradients, is_training must be set to True.")
    tf.cudnnRNNBackprop(input = op.inputs[0],
                        inputH = op.inputs[1],
                        inputC = op.inputs[2],
                        params = op.inputs[3],
                        output = op.outputs[0],
                        outputH = op.outputs[1],
                        outputC = op.outputs[2],
                        outputBackprop = grads[0],
                        outputHBackprop = grads[1],
                        outputCBackprop = grads[2],
                        reserveSpace = op.outputs[3],
                        dropout = op.attrFloat("dropout"),
                        seed = op.attrLong("seed"),
                        seed2 = op.attrLong("seed2"),
                        rnnMode = op.attrString("rnn_mode"),
                        inputMode = op.attrString("input_mode"),
                        direction = op.attrString("direction"))  //return@register
  }
  register("CudnnRNNV2") { op, grad ->
    val grads = grad.map { it!!.toOutput() }
    if (!op.attrBool("is_training"))
      error("To use CudnnRNNV2 in gradients, is_training must be set to True.")
    tf.cudnnRNNBackpropV2(input = op.inputs[0],
                          inputH = op.inputs[1],
                          inputC = op.inputs[2],
                          params = op.inputs[3],
                          output = op.outputs[0],
                          outputH = op.outputs[1],
                          outputC = op.outputs[2],
                          outputBackprop = grads[0],
                          outputHBackprop = grads[1],
                          outputCBackprop = grads[2],
                          reserveSpace = op.outputs[3],
                          hostReserved = op.outputs[4],
                          dropout = op.attrFloat("dropout"),
                          seed = op.attrLong("seed"),
                          seed2 = op.attrLong("seed2"),
                          rnnMode = op.attrString("rnn_mode"),
                          inputMode = op.attrString("input_mode"),
                          direction = op.attrString("direction"))  //return@register
  }
}