/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.NameAttrList
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors

fun TF.batch(in_tensors: Output, num_batch_threads: Long, max_batch_size: Long, batch_timeout_micros: Long, grad_timeout_micros: Long, max_enqueued_batches: Long = 10L, allowed_batch_sizes: Array<Long> = arrayOf(), container: String = "", shared_name: String = "", batching_queue: String = "", name: String = "Batch") = run {
  buildOpTensors("Batch", name) {
    addInput(in_tensors, false)
    attr("num_batch_threads", num_batch_threads)
    attr("max_batch_size", max_batch_size)
    attr("batch_timeout_micros", batch_timeout_micros)
    attr("grad_timeout_micros", grad_timeout_micros)
    attr("max_enqueued_batches", max_enqueued_batches)
    attr("allowed_batch_sizes", allowed_batch_sizes)
    attr("container", container)
    attr("shared_name", shared_name)
    attr("batching_queue", batching_queue)
  }
}

fun TF.batchFunction(in_tensors: Output, captured_tensors: Output, f: NameAttrList, num_batch_threads: Long, max_batch_size: Long, batch_timeout_micros: Long, tout: Array<Long>, max_enqueued_batches: Long = 10L, allowed_batch_sizes: Array<Long> = arrayOf(), container: String = "", shared_name: String = "", batching_queue: String = "", name: String = "BatchFunction") = run {
  buildOpTensors("BatchFunction", name) {
    addInput(in_tensors, false)
    addInput(captured_tensors, false)
    attr("f", f)
    attr("num_batch_threads", num_batch_threads)
    attr("max_batch_size", max_batch_size)
    attr("batch_timeout_micros", batch_timeout_micros)
    attr("Tout", tout)
    attr("max_enqueued_batches", max_enqueued_batches)
    attr("allowed_batch_sizes", allowed_batch_sizes)
    attr("container", container)
    attr("shared_name", shared_name)
    attr("batching_queue", batching_queue)
  }
}

fun TF.unbatch(batched_tensor: Output, batch_index: Output, id: Output, timeout_micros: Long, container: String = "", shared_name: String = "", name: String = "Unbatch") = run {
  buildOpTensor("Unbatch", name) {
    addInput(batched_tensor, false)
    addInput(batch_index, false)
    addInput(id, false)
    attr("timeout_micros", timeout_micros)
    attr("container", container)
    attr("shared_name", shared_name)
  }
}

fun TF.unbatchGrad(original_input: Output, batch_index: Output, grad: Output, id: Output, container: String = "", shared_name: String = "", name: String = "UnbatchGrad") = run {
  buildOpTensor("UnbatchGrad", name) {
    addInput(original_input, false)
    addInput(batch_index, false)
    addInput(grad, false)
    addInput(id, false)
    attr("container", container)
    attr("shared_name", shared_name)
  }
}
