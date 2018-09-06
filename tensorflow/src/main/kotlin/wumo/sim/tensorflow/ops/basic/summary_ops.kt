package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_summary_ops

object summary_ops {
  interface API {
    fun closeSummaryWriter(writer: Output, name: String = "CloseSummaryWriter"): Op {
      return gen_summary_ops.closeSummaryWriter(writer, name)
    }
    
    fun createSummaryDbWriter(writer: Output, dbUri: Output, experimentName: Output, runName: Output, userName: Output, name: String = "CreateSummaryDbWriter"): Op {
      return gen_summary_ops.createSummaryDbWriter(writer, dbUri, experimentName, runName, userName, name)
    }
    
    fun createSummaryFileWriter(writer: Output, logdir: Output, maxQueue: Output, flushMillis: Output, filenameSuffix: Output, name: String = "CreateSummaryFileWriter"): Op {
      return gen_summary_ops.createSummaryFileWriter(writer, logdir, maxQueue, flushMillis, filenameSuffix, name)
    }
    
    fun flushSummaryWriter(writer: Output, name: String = "FlushSummaryWriter"): Op {
      return gen_summary_ops.flushSummaryWriter(writer, name)
    }
    
    fun importEvent(writer: Output, event: Output, name: String = "ImportEvent"): Op {
      return gen_summary_ops.importEvent(writer, event, name)
    }
    
    fun summaryWriter(sharedName: String = "", container: String = "", name: String = "SummaryWriter"): Output {
      return gen_summary_ops.summaryWriter(sharedName, container, name)
    }
    
    fun writeAudioSummary(writer: Output, step: Output, tag: Output, tensor: Output, sampleRate: Output, maxOutputs: Long = 3L, name: String = "WriteAudioSummary"): Op {
      return gen_summary_ops.writeAudioSummary(writer, step, tag, tensor, sampleRate, maxOutputs, name)
    }
    
    fun writeGraphSummary(writer: Output, step: Output, tensor: Output, name: String = "WriteGraphSummary"): Op {
      return gen_summary_ops.writeGraphSummary(writer, step, tensor, name)
    }
    
    fun writeHistogramSummary(writer: Output, step: Output, tag: Output, values: Output, name: String = "WriteHistogramSummary"): Op {
      return gen_summary_ops.writeHistogramSummary(writer, step, tag, values, name)
    }
    
    fun writeImageSummary(writer: Output, step: Output, tag: Output, tensor: Output, badColor: Output, maxImages: Long = 3L, name: String = "WriteImageSummary"): Op {
      return gen_summary_ops.writeImageSummary(writer, step, tag, tensor, badColor, maxImages, name)
    }
    
    fun writeScalarSummary(writer: Output, step: Output, tag: Output, value: Output, name: String = "WriteScalarSummary"): Op {
      return gen_summary_ops.writeScalarSummary(writer, step, tag, value, name)
    }
    
    fun writeSummary(writer: Output, step: Output, tensor: Output, tag: Output, summaryMetadata: Output, name: String = "WriteSummary"): Op {
      return gen_summary_ops.writeSummary(writer, step, tensor, tag, summaryMetadata, name)
    }
  }
}