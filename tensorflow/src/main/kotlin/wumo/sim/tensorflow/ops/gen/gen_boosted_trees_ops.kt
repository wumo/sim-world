/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output

interface gen_boosted_trees_ops {
  fun boostedTreesCalculateBestGainsPerFeature(node_id_range: Output, stats_summary_list: List<Output>, l1: Output, l2: Output, tree_complexity: Output, min_node_weight: Output, max_splits: Long, name: String = "BoostedTreesCalculateBestGainsPerFeature") = run {
    buildOpTensors("BoostedTreesCalculateBestGainsPerFeature", name) {
      addInput(node_id_range, false)
      addInput(stats_summary_list, false)
      addInput(l1, false)
      addInput(l2, false)
      addInput(tree_complexity, false)
      addInput(min_node_weight, false)
      attr("max_splits", max_splits)
    }
  }
  
  fun boostedTreesCenterBias(tree_ensemble_handle: Output, mean_gradients: Output, mean_hessians: Output, l1: Output, l2: Output, name: String = "BoostedTreesCenterBias") = run {
    buildOpTensor("BoostedTreesCenterBias", name) {
      addInput(tree_ensemble_handle, false)
      addInput(mean_gradients, false)
      addInput(mean_hessians, false)
      addInput(l1, false)
      addInput(l2, false)
    }
  }
  
  fun boostedTreesCreateEnsemble(tree_ensemble_handle: Output, stamp_token: Output, tree_ensemble_serialized: Output, name: String = "BoostedTreesCreateEnsemble") = run {
    buildOp("BoostedTreesCreateEnsemble", name) {
      addInput(tree_ensemble_handle, false)
      addInput(stamp_token, false)
      addInput(tree_ensemble_serialized, false)
    }
  }
  
  fun boostedTreesDeserializeEnsemble(tree_ensemble_handle: Output, stamp_token: Output, tree_ensemble_serialized: Output, name: String = "BoostedTreesDeserializeEnsemble") = run {
    buildOp("BoostedTreesDeserializeEnsemble", name) {
      addInput(tree_ensemble_handle, false)
      addInput(stamp_token, false)
      addInput(tree_ensemble_serialized, false)
    }
  }
  
  fun boostedTreesEnsembleResourceHandleOp(container: String = "", shared_name: String = "", name: String = "BoostedTreesEnsembleResourceHandleOp") = run {
    buildOpTensor("BoostedTreesEnsembleResourceHandleOp", name) {
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
  
  fun boostedTreesExampleDebugOutputs(tree_ensemble_handle: Output, bucketized_features: List<Output>, logits_dimension: Long, name: String = "BoostedTreesExampleDebugOutputs") = run {
    buildOpTensor("BoostedTreesExampleDebugOutputs", name) {
      addInput(tree_ensemble_handle, false)
      addInput(bucketized_features, false)
      attr("logits_dimension", logits_dimension)
    }
  }
  
  fun boostedTreesGetEnsembleStates(tree_ensemble_handle: Output, name: String = "BoostedTreesGetEnsembleStates") = run {
    buildOpTensors("BoostedTreesGetEnsembleStates", name) {
      addInput(tree_ensemble_handle, false)
    }
  }
  
  fun boostedTreesMakeStatsSummary(node_ids: Output, gradients: Output, hessians: Output, bucketized_features_list: List<Output>, max_splits: Long, num_buckets: Long, name: String = "BoostedTreesMakeStatsSummary") = run {
    buildOpTensor("BoostedTreesMakeStatsSummary", name) {
      addInput(node_ids, false)
      addInput(gradients, false)
      addInput(hessians, false)
      addInput(bucketized_features_list, false)
      attr("max_splits", max_splits)
      attr("num_buckets", num_buckets)
    }
  }
  
  fun boostedTreesPredict(tree_ensemble_handle: Output, bucketized_features: List<Output>, logits_dimension: Long, name: String = "BoostedTreesPredict") = run {
    buildOpTensor("BoostedTreesPredict", name) {
      addInput(tree_ensemble_handle, false)
      addInput(bucketized_features, false)
      attr("logits_dimension", logits_dimension)
    }
  }
  
  fun boostedTreesSerializeEnsemble(tree_ensemble_handle: Output, name: String = "BoostedTreesSerializeEnsemble") = run {
    buildOpTensors("BoostedTreesSerializeEnsemble", name) {
      addInput(tree_ensemble_handle, false)
    }
  }
  
  fun boostedTreesTrainingPredict(tree_ensemble_handle: Output, cached_tree_ids: Output, cached_node_ids: Output, bucketized_features: List<Output>, logits_dimension: Long, name: String = "BoostedTreesTrainingPredict") = run {
    buildOpTensors("BoostedTreesTrainingPredict", name) {
      addInput(tree_ensemble_handle, false)
      addInput(cached_tree_ids, false)
      addInput(cached_node_ids, false)
      addInput(bucketized_features, false)
      attr("logits_dimension", logits_dimension)
    }
  }
  
  fun boostedTreesUpdateEnsemble(tree_ensemble_handle: Output, feature_ids: Output, node_ids: List<Output>, gains: List<Output>, thresholds: List<Output>, left_node_contribs: List<Output>, right_node_contribs: List<Output>, max_depth: Output, learning_rate: Output, pruning_mode: Long, name: String = "BoostedTreesUpdateEnsemble") = run {
    buildOp("BoostedTreesUpdateEnsemble", name) {
      addInput(tree_ensemble_handle, false)
      addInput(feature_ids, false)
      addInput(node_ids, false)
      addInput(gains, false)
      addInput(thresholds, false)
      addInput(left_node_contribs, false)
      addInput(right_node_contribs, false)
      addInput(max_depth, false)
      addInput(learning_rate, false)
      attr("pruning_mode", pruning_mode)
    }
  }
  
  fun isBoostedTreesEnsembleInitialized(tree_ensemble_handle: Output, name: String = "IsBoostedTreesEnsembleInitialized") = run {
    buildOpTensor("IsBoostedTreesEnsembleInitialized", name) {
      addInput(tree_ensemble_handle, false)
    }
  }
}