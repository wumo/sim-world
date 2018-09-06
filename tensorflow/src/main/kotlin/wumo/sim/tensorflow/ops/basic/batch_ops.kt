package wumo.sim.tensorflow.ops.basic

import org.bytedeco.javacpp.tensorflow.NameAttrList
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_batch_ops

object batch_ops {
  interface API {
    fun batch(inTensors: Output, numBatchThreads: Long, maxBatchSize: Long, batchTimeoutMicros: Long, gradTimeoutMicros: Long, maxEnqueuedBatches: Long = 10L, allowedBatchSizes: Array<Long> = arrayOf(), container: String = "", sharedName: String = "", batchingQueue: String = "", name: String = "Batch"): List<Output> {
      return gen_batch_ops.batch(inTensors, numBatchThreads, maxBatchSize, batchTimeoutMicros, gradTimeoutMicros, maxEnqueuedBatches, allowedBatchSizes, container, sharedName, batchingQueue, name)
    }
    
    fun batchFunction(inTensors: Output, capturedTensors: Output, f: NameAttrList, numBatchThreads: Long, maxBatchSize: Long, batchTimeoutMicros: Long, tout: Array<Long>, maxEnqueuedBatches: Long = 10L, allowedBatchSizes: Array<Long> = arrayOf(), container: String = "", sharedName: String = "", batchingQueue: String = "", name: String = "BatchFunction"): List<Output> {
      return gen_batch_ops.batchFunction(inTensors, capturedTensors, f, numBatchThreads, maxBatchSize, batchTimeoutMicros, tout, maxEnqueuedBatches, allowedBatchSizes, container, sharedName, batchingQueue, name)
    }
    
    fun unbatch(batchedTensor: Output, batchIndex: Output, id: Output, timeoutMicros: Long, container: String = "", sharedName: String = "", name: String = "Unbatch"): Output {
      return gen_batch_ops.unbatch(batchedTensor, batchIndex, id, timeoutMicros, container, sharedName, name)
    }
    
    fun unbatchGrad(originalInput: Output, batchIndex: Output, grad: Output, id: Output, container: String = "", sharedName: String = "", name: String = "UnbatchGrad"): Output {
      return gen_batch_ops.unbatchGrad(originalInput, batchIndex, grad, id, container, sharedName, name)
    }
  }
}