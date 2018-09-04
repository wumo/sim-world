import wumo.sim.tensorflow.ops.IndexedSlices
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike
import wumo.sim.tensorflow.ops.get
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.INT32

fun register_data_flow_grad() {
  /**Gradients for operators defined in data_flow_ops.py.*/
/* from__future__importabsolute_import */
/* from__future__importdivision */
/* from__future__importprint_function */
/* fromsix.movesimportxrange */
/* fromtensorflow.python.frameworkimportdtypes */
/* fromtensorflow.python.frameworkimportops */
/* fromtensorflow.python.opsimportarray_ops */
/* fromtensorflow.python.opsimportdata_flow_ops */
/* fromtensorflow.python.opsimportmath_ops */
  register("DynamicPartition") { op, grads ->
    val grads = grads.map { it!!.toOutput() }
    /**Gradients for DynamicPartition.*/
    val data = op.inputs[0]
    val indices = op.inputs[1]
    val numPartitions = op.attrLong("num_partitions")
    val prefixShape = tf.shape(indices)
    val originalIndices = tf.reshape(tf.range(tf.const(0), tf.prod(prefixShape)), prefixShape)
    val partitionedIndices = tf.dynamicPartition(originalIndices, indices, numPartitions)
    var reconstructed = tf.dynamicStitch(partitionedIndices, grads)
    reconstructed = tf.reshape(reconstructed, tf.shape(data))
    listOf(reconstructed, null) //return@register
  }
  register("DynamicStitch", "ParallelDynamicStitch") { op, grad ->
    var grad = grad[0]!!
    /**Gradients for DynamicStitch and ParallelDynamicStitch.*/
    val numValues = (op.inputs).size / 2
    val indicesGrad = List<OutputLike?>(numValues) { null }
    
    val inputs = op.inputs.take(numValues).map { tf.cast(it, INT32) }
    if (grad is IndexedSlices) {
      val outputShape = tf.shape(op.outputs[0])
      val outputRows = outputShape[0]
      grad = tf.unsortedSegmentSum(grad.values, grad.indices, outputRows)
    }
    grad as Output
    val valuesGrad = inputs.map { tf.gather(grad, it) }
    indicesGrad + valuesGrad  //return@register
  }
  registerNonDifferentiable("Queue")
  registerNonDifferentiable("QueueEnqueue")
  registerNonDifferentiable("QueueEnqueueMany")
  registerNonDifferentiable("QueueDequeue")
  registerNonDifferentiable("QueueDequeueMany")
  registerNonDifferentiable("QueueDequeueUpTo")
  registerNonDifferentiable("QueueClose")
  registerNonDifferentiable("QueueSize")
  registerNonDifferentiable("Stack")
  registerNonDifferentiable("StackPush")
  registerNonDifferentiable("StackPop")
  registerNonDifferentiable("StackClose")
  registerNonDifferentiable("GetSessionHandle")
  registerNonDifferentiable("GetSessionHandleV2")
  registerNonDifferentiable("GetSessionTensor")
  registerNonDifferentiable("DeleteSessionTensor")
}