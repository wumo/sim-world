/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output

object gen_candidate_sampling_ops {
  fun allCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, seed: Long = 0L, seed2: Long = 0L, name: String = "AllCandidateSampler"): List<Output> =
      buildOpTensors("AllCandidateSampler", name) {
        addInput(trueClasses, false)
        attr("num_true", numTrue)
        attr("num_sampled", numSampled)
        attr("unique", unique)
        attr("seed", seed)
        attr("seed2", seed2)
      }
  
  fun computeAccidentalHits(trueClasses: Output, sampledCandidates: Output, numTrue: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "ComputeAccidentalHits"): List<Output> =
      buildOpTensors("ComputeAccidentalHits", name) {
        addInput(trueClasses, false)
        addInput(sampledCandidates, false)
        attr("num_true", numTrue)
        attr("seed", seed)
        attr("seed2", seed2)
      }
  
  fun fixedUnigramCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, vocabFile: String = "", distortion: Float = 1.0f, numReservedIds: Long = 0L, numShards: Long = 1L, shard: Long = 0L, unigrams: Array<Float> = arrayOf(), seed: Long = 0L, seed2: Long = 0L, name: String = "FixedUnigramCandidateSampler"): List<Output> =
      buildOpTensors("FixedUnigramCandidateSampler", name) {
        addInput(trueClasses, false)
        attr("num_true", numTrue)
        attr("num_sampled", numSampled)
        attr("unique", unique)
        attr("range_max", rangeMax)
        attr("vocab_file", vocabFile)
        attr("distortion", distortion)
        attr("num_reserved_ids", numReservedIds)
        attr("num_shards", numShards)
        attr("shard", shard)
        attr("unigrams", unigrams)
        attr("seed", seed)
        attr("seed2", seed2)
      }
  
  fun learnedUnigramCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "LearnedUnigramCandidateSampler"): List<Output> =
      buildOpTensors("LearnedUnigramCandidateSampler", name) {
        addInput(trueClasses, false)
        attr("num_true", numTrue)
        attr("num_sampled", numSampled)
        attr("unique", unique)
        attr("range_max", rangeMax)
        attr("seed", seed)
        attr("seed2", seed2)
      }
  
  fun logUniformCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "LogUniformCandidateSampler"): List<Output> =
      buildOpTensors("LogUniformCandidateSampler", name) {
        addInput(trueClasses, false)
        attr("num_true", numTrue)
        attr("num_sampled", numSampled)
        attr("unique", unique)
        attr("range_max", rangeMax)
        attr("seed", seed)
        attr("seed2", seed2)
      }
  
  fun threadUnsafeUnigramCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "ThreadUnsafeUnigramCandidateSampler"): List<Output> =
      buildOpTensors("ThreadUnsafeUnigramCandidateSampler", name) {
        addInput(trueClasses, false)
        attr("num_true", numTrue)
        attr("num_sampled", numSampled)
        attr("unique", unique)
        attr("range_max", rangeMax)
        attr("seed", seed)
        attr("seed2", seed2)
      }
  
  fun uniformCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "UniformCandidateSampler"): List<Output> =
      buildOpTensors("UniformCandidateSampler", name) {
        addInput(trueClasses, false)
        attr("num_true", numTrue)
        attr("num_sampled", numSampled)
        attr("unique", unique)
        attr("range_max", rangeMax)
        attr("seed", seed)
        attr("seed2", seed2)
      }
}