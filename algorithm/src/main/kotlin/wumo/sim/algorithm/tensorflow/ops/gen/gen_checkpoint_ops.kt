/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.algorithm.tensorflow.ops.gen

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.buildOpTensor
import wumo.sim.algorithm.tensorflow.buildOpTensors

fun TF.generateVocabRemapping(new_vocab_file: Tensor, old_vocab_file: Tensor, new_vocab_offset: Long, num_new_vocab: Long, old_vocab_size: Long = -1L, name: String = "GenerateVocabRemapping") = run {
  buildOpTensors("GenerateVocabRemapping", name) {
    addInput(new_vocab_file, false)
    addInput(old_vocab_file, false)
    attr("new_vocab_offset", new_vocab_offset)
    attr("num_new_vocab", num_new_vocab)
    attr("old_vocab_size", old_vocab_size)
  }
}

fun TF.loadAndRemapMatrix(ckpt_path: Tensor, old_tensor_name: Tensor, row_remapping: Tensor, col_remapping: Tensor, initializing_values: Tensor, num_rows: Long, num_cols: Long, max_rows_in_memory: Long = -1L, name: String = "LoadAndRemapMatrix") = run {
  buildOpTensor("LoadAndRemapMatrix", name) {
    addInput(ckpt_path, false)
    addInput(old_tensor_name, false)
    addInput(row_remapping, false)
    addInput(col_remapping, false)
    addInput(initializing_values, false)
    attr("num_rows", num_rows)
    attr("num_cols", num_cols)
    attr("max_rows_in_memory", max_rows_in_memory)
  }
}
