package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_candidate_sampling_ops

object candidate_sampling_ops {
  interface API {
    fun allCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, seed: Long = 0L, seed2: Long = 0L, name: String = "AllCandidateSampler"): List<Output> {
      return gen_candidate_sampling_ops.allCandidateSampler(trueClasses, numTrue, numSampled, unique, seed, seed2, name)
    }
    
    fun computeAccidentalHits(trueClasses: Output, sampledCandidates: Output, numTrue: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "ComputeAccidentalHits"): List<Output> {
      return gen_candidate_sampling_ops.computeAccidentalHits(trueClasses, sampledCandidates, numTrue, seed, seed2, name)
    }
    
    fun fixedUnigramCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, vocabFile: String = "", distortion: Float = 1.0f, numReservedIds: Long = 0L, numShards: Long = 1L, shard: Long = 0L, unigrams: Array<Float> = arrayOf(), seed: Long = 0L, seed2: Long = 0L, name: String = "FixedUnigramCandidateSampler"): List<Output> {
      return gen_candidate_sampling_ops.fixedUnigramCandidateSampler(trueClasses, numTrue, numSampled, unique, rangeMax, vocabFile, distortion, numReservedIds, numShards, shard, unigrams, seed, seed2, name)
    }
    
    fun learnedUnigramCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "LearnedUnigramCandidateSampler"): List<Output> {
      return gen_candidate_sampling_ops.learnedUnigramCandidateSampler(trueClasses, numTrue, numSampled, unique, rangeMax, seed, seed2, name)
    }
    
    fun logUniformCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "LogUniformCandidateSampler"): List<Output> {
      return gen_candidate_sampling_ops.logUniformCandidateSampler(trueClasses, numTrue, numSampled, unique, rangeMax, seed, seed2, name)
    }
    
    fun threadUnsafeUnigramCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "ThreadUnsafeUnigramCandidateSampler"): List<Output> {
      return gen_candidate_sampling_ops.threadUnsafeUnigramCandidateSampler(trueClasses, numTrue, numSampled, unique, rangeMax, seed, seed2, name)
    }
    
    fun uniformCandidateSampler(trueClasses: Output, numTrue: Long, numSampled: Long, unique: Boolean, rangeMax: Long, seed: Long = 0L, seed2: Long = 0L, name: String = "UniformCandidateSampler"): List<Output> {
      return gen_candidate_sampling_ops.uniformCandidateSampler(trueClasses, numTrue, numSampled, unique, rangeMax, seed, seed2, name)
    }
  }
}