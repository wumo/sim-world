package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_logging_ops
import wumo.sim.util.ndarray.NDArray

object logging_ops {
  interface API {
    fun assert(condition: Output, data: Output, summarize: Long = 3L, name: String = "Assert"): Op {
      return gen_logging_ops.assert(condition, data, summarize, name)
    }
    
    fun audioSummary(tag: Output, tensor: Output, sampleRate: Float, maxOutputs: Long = 3L, name: String = "AudioSummary"): Output {
      return gen_logging_ops.audioSummary(tag, tensor, sampleRate, maxOutputs, name)
    }
    
    fun audioSummaryV2(tag: Output, tensor: Output, sampleRate: Output, maxOutputs: Long = 3L, name: String = "AudioSummaryV2"): Output {
      return gen_logging_ops.audioSummaryV2(tag, tensor, sampleRate, maxOutputs, name)
    }
    
    fun histogramSummary(tag: Output, values: Output, name: String = "HistogramSummary"): Output {
      return gen_logging_ops.histogramSummary(tag, values, name)
    }
    
    fun imageSummary(tag: Output, tensor: Output, maxImages: Long = 3L, badColor: NDArray<*>, name: String = "ImageSummary"): Output {
      return gen_logging_ops.imageSummary(tag, tensor, maxImages, badColor, name)
    }
    
    fun mergeSummary(inputs: List<Output>, name: String = "MergeSummary"): Output {
      return gen_logging_ops.mergeSummary(inputs, name)
    }
    
    fun print(input: Output, data: Output, message: String = "", firstN: Long = -1L, summarize: Long = 3L, name: String = "Print"): Output {
      return gen_logging_ops.print(input, data, message, firstN, summarize, name)
    }
    
    fun scalarSummary(tags: Output, values: Output, name: String = "ScalarSummary"): Output {
      return gen_logging_ops.scalarSummary(tags, values, name)
    }
    
    fun tensorSummary(tensor: Output, description: String = "", labels: Array<String> = arrayOf(), displayName: String = "", name: String = "TensorSummary"): Output {
      return gen_logging_ops.tensorSummary(tensor, description, labels, displayName, name)
    }
    
    fun tensorSummaryV2(tag: Output, tensor: Output, serializedSummaryMetadata: Output, name: String = "TensorSummaryV2"): Output {
      return gen_logging_ops.tensorSummaryV2(tag, tensor, serializedSummaryMetadata, name)
    }
    
    fun timestamp(name: String = "Timestamp"): Output {
      return gen_logging_ops.timestamp(name)
    }
  }
}