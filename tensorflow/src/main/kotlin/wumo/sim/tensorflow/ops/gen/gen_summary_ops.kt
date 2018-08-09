/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor

fun TF.closeSummaryWriter(writer: Output, name: String = "CloseSummaryWriter") = run {
  buildOp("CloseSummaryWriter", name) {
    addInput(writer, false)
  }
}

fun TF.createSummaryDbWriter(writer: Output, db_uri: Output, experiment_name: Output, run_name: Output, user_name: Output, name: String = "CreateSummaryDbWriter") = run {
  buildOp("CreateSummaryDbWriter", name) {
    addInput(writer, false)
    addInput(db_uri, false)
    addInput(experiment_name, false)
    addInput(run_name, false)
    addInput(user_name, false)
  }
}

fun TF.createSummaryFileWriter(writer: Output, logdir: Output, max_queue: Output, flush_millis: Output, filename_suffix: Output, name: String = "CreateSummaryFileWriter") = run {
  buildOp("CreateSummaryFileWriter", name) {
    addInput(writer, false)
    addInput(logdir, false)
    addInput(max_queue, false)
    addInput(flush_millis, false)
    addInput(filename_suffix, false)
  }
}

fun TF.flushSummaryWriter(writer: Output, name: String = "FlushSummaryWriter") = run {
  buildOp("FlushSummaryWriter", name) {
    addInput(writer, false)
  }
}

fun TF.importEvent(writer: Output, event: Output, name: String = "ImportEvent") = run {
  buildOp("ImportEvent", name) {
    addInput(writer, false)
    addInput(event, false)
  }
}

fun TF.summaryWriter(shared_name: String = "", container: String = "", name: String = "SummaryWriter") = run {
  buildOpTensor("SummaryWriter", name) {
    attr("shared_name", shared_name)
    attr("container", container)
  }
}

fun TF.writeAudioSummary(writer: Output, step: Output, tag: Output, tensor: Output, sample_rate: Output, max_outputs: Long = 3L, name: String = "WriteAudioSummary") = run {
  buildOp("WriteAudioSummary", name) {
    addInput(writer, false)
    addInput(step, false)
    addInput(tag, false)
    addInput(tensor, false)
    addInput(sample_rate, false)
    attr("max_outputs", max_outputs)
  }
}

fun TF.writeGraphSummary(writer: Output, step: Output, tensor: Output, name: String = "WriteGraphSummary") = run {
  buildOp("WriteGraphSummary", name) {
    addInput(writer, false)
    addInput(step, false)
    addInput(tensor, false)
  }
}

fun TF.writeHistogramSummary(writer: Output, step: Output, tag: Output, values: Output, name: String = "WriteHistogramSummary") = run {
  buildOp("WriteHistogramSummary", name) {
    addInput(writer, false)
    addInput(step, false)
    addInput(tag, false)
    addInput(values, false)
  }
}

fun TF.writeImageSummary(writer: Output, step: Output, tag: Output, tensor: Output, bad_color: Output, max_images: Long = 3L, name: String = "WriteImageSummary") = run {
  buildOp("WriteImageSummary", name) {
    addInput(writer, false)
    addInput(step, false)
    addInput(tag, false)
    addInput(tensor, false)
    addInput(bad_color, false)
    attr("max_images", max_images)
  }
}

fun TF.writeScalarSummary(writer: Output, step: Output, tag: Output, value: Output, name: String = "WriteScalarSummary") = run {
  buildOp("WriteScalarSummary", name) {
    addInput(writer, false)
    addInput(step, false)
    addInput(tag, false)
    addInput(value, false)
  }
}

fun TF.writeSummary(writer: Output, step: Output, tensor: Output, tag: Output, summary_metadata: Output, name: String = "WriteSummary") = run {
  buildOp("WriteSummary", name) {
    addInput(writer, false)
    addInput(step, false)
    addInput(tensor, false)
    addInput(tag, false)
    addInput(summary_metadata, false)
  }
}
