package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_ctc_ops

object ctc_ops {
  interface API {
    fun cTCBeamSearchDecoder(inputs: Output, sequenceLength: Output, beamWidth: Long, topPaths: Long, mergeRepeated: Boolean = true, name: String = "CTCBeamSearchDecoder"): List<Output> {
      return gen_ctc_ops.cTCBeamSearchDecoder(inputs, sequenceLength, beamWidth, topPaths, mergeRepeated, name)
    }
    
    fun cTCGreedyDecoder(inputs: Output, sequenceLength: Output, mergeRepeated: Boolean = false, name: String = "CTCGreedyDecoder"): List<Output> {
      return gen_ctc_ops.cTCGreedyDecoder(inputs, sequenceLength, mergeRepeated, name)
    }
    
    fun cTCLoss(inputs: Output, labelsIndices: Output, labelsValues: Output, sequenceLength: Output, preprocessCollapseRepeated: Boolean = false, ctcMergeRepeated: Boolean = true, ignoreLongerOutputsThanInputs: Boolean = false, name: String = "CTCLoss"): List<Output> {
      return gen_ctc_ops.cTCLoss(inputs, labelsIndices, labelsValues, sequenceLength, preprocessCollapseRepeated, ctcMergeRepeated, ignoreLongerOutputsThanInputs, name)
    }
  }
}