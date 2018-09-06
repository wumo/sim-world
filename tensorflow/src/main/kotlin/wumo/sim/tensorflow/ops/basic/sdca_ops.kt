package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_sdca_ops

object sdca_ops {
  interface API {
    fun sdcaFprint(input: Output, name: String = "SdcaFprint"): Output {
      return gen_sdca_ops.sdcaFprint(input, name)
    }
    
    fun sdcaOptimizer(sparseExampleIndices: List<Output>, sparseFeatureIndices: List<Output>, sparseFeatureValues: List<Output>, denseFeatures: List<Output>, exampleWeights: Output, exampleLabels: Output, sparseIndices: List<Output>, sparseWeights: List<Output>, denseWeights: List<Output>, exampleStateData: Output, lossType: String, l1: Float, l2: Float, numLossPartitions: Long, numInnerIterations: Long, adaptative: Boolean = false, name: String = "SdcaOptimizer"): List<Output> {
      return gen_sdca_ops.sdcaOptimizer(sparseExampleIndices, sparseFeatureIndices, sparseFeatureValues, denseFeatures, exampleWeights, exampleLabels, sparseIndices, sparseWeights, denseWeights, exampleStateData, lossType, l1, l2, numLossPartitions, numInnerIterations, adaptative, name)
    }
    
    fun sdcaShrinkL1(weights: List<Output>, l1: Float, l2: Float, name: String = "SdcaShrinkL1"): Op {
      return gen_sdca_ops.sdcaShrinkL1(weights, l1, l2, name)
    }
  }
}