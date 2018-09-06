package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_boosted_trees_ops

object boosted_trees_ops {
  interface API {
    fun boostedTreesCalculateBestGainsPerFeature(nodeIdRange: Output, statsSummaryList: List<Output>, l1: Output, l2: Output, treeComplexity: Output, minNodeWeight: Output, maxSplits: Long, name: String = "BoostedTreesCalculateBestGainsPerFeature"): List<Output> {
      return gen_boosted_trees_ops.boostedTreesCalculateBestGainsPerFeature(nodeIdRange, statsSummaryList, l1, l2, treeComplexity, minNodeWeight, maxSplits, name)
    }
    
    fun boostedTreesCenterBias(treeEnsembleHandle: Output, meanGradients: Output, meanHessians: Output, l1: Output, l2: Output, name: String = "BoostedTreesCenterBias"): Output {
      return gen_boosted_trees_ops.boostedTreesCenterBias(treeEnsembleHandle, meanGradients, meanHessians, l1, l2, name)
    }
    
    fun boostedTreesCreateEnsemble(treeEnsembleHandle: Output, stampToken: Output, treeEnsembleSerialized: Output, name: String = "BoostedTreesCreateEnsemble"): Op {
      return gen_boosted_trees_ops.boostedTreesCreateEnsemble(treeEnsembleHandle, stampToken, treeEnsembleSerialized, name)
    }
    
    fun boostedTreesDeserializeEnsemble(treeEnsembleHandle: Output, stampToken: Output, treeEnsembleSerialized: Output, name: String = "BoostedTreesDeserializeEnsemble"): Op {
      return gen_boosted_trees_ops.boostedTreesDeserializeEnsemble(treeEnsembleHandle, stampToken, treeEnsembleSerialized, name)
    }
    
    fun boostedTreesEnsembleResourceHandleOp(container: String = "", sharedName: String = "", name: String = "BoostedTreesEnsembleResourceHandleOp"): Output {
      return gen_boosted_trees_ops.boostedTreesEnsembleResourceHandleOp(container, sharedName, name)
    }
    
    fun boostedTreesExampleDebugOutputs(treeEnsembleHandle: Output, bucketizedFeatures: List<Output>, logitsDimension: Long, name: String = "BoostedTreesExampleDebugOutputs"): Output {
      return gen_boosted_trees_ops.boostedTreesExampleDebugOutputs(treeEnsembleHandle, bucketizedFeatures, logitsDimension, name)
    }
    
    fun boostedTreesGetEnsembleStates(treeEnsembleHandle: Output, name: String = "BoostedTreesGetEnsembleStates"): List<Output> {
      return gen_boosted_trees_ops.boostedTreesGetEnsembleStates(treeEnsembleHandle, name)
    }
    
    fun boostedTreesMakeStatsSummary(nodeIds: Output, gradients: Output, hessians: Output, bucketizedFeaturesList: List<Output>, maxSplits: Long, numBuckets: Long, name: String = "BoostedTreesMakeStatsSummary"): Output {
      return gen_boosted_trees_ops.boostedTreesMakeStatsSummary(nodeIds, gradients, hessians, bucketizedFeaturesList, maxSplits, numBuckets, name)
    }
    
    fun boostedTreesPredict(treeEnsembleHandle: Output, bucketizedFeatures: List<Output>, logitsDimension: Long, name: String = "BoostedTreesPredict"): Output {
      return gen_boosted_trees_ops.boostedTreesPredict(treeEnsembleHandle, bucketizedFeatures, logitsDimension, name)
    }
    
    fun boostedTreesSerializeEnsemble(treeEnsembleHandle: Output, name: String = "BoostedTreesSerializeEnsemble"): List<Output> {
      return gen_boosted_trees_ops.boostedTreesSerializeEnsemble(treeEnsembleHandle, name)
    }
    
    fun boostedTreesTrainingPredict(treeEnsembleHandle: Output, cachedTreeIds: Output, cachedNodeIds: Output, bucketizedFeatures: List<Output>, logitsDimension: Long, name: String = "BoostedTreesTrainingPredict"): List<Output> {
      return gen_boosted_trees_ops.boostedTreesTrainingPredict(treeEnsembleHandle, cachedTreeIds, cachedNodeIds, bucketizedFeatures, logitsDimension, name)
    }
    
    fun boostedTreesUpdateEnsemble(treeEnsembleHandle: Output, featureIds: Output, nodeIds: List<Output>, gains: List<Output>, thresholds: List<Output>, leftNodeContribs: List<Output>, rightNodeContribs: List<Output>, maxDepth: Output, learningRate: Output, pruningMode: Long, name: String = "BoostedTreesUpdateEnsemble"): Op {
      return gen_boosted_trees_ops.boostedTreesUpdateEnsemble(treeEnsembleHandle, featureIds, nodeIds, gains, thresholds, leftNodeContribs, rightNodeContribs, maxDepth, learningRate, pruningMode, name)
    }
    
    fun isBoostedTreesEnsembleInitialized(treeEnsembleHandle: Output, name: String = "IsBoostedTreesEnsembleInitialized"): Output {
      return gen_boosted_trees_ops.isBoostedTreesEnsembleInitialized(treeEnsembleHandle, name)
    }
  }
}