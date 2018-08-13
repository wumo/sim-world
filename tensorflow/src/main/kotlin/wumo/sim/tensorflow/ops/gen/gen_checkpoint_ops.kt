/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.ops.Output
import wumo.sim.util.Shape
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.tf
import wumo.sim.util.ndarray.NDArray

interface gen_checkpoint_ops {
  fun _generateVocabRemapping(new_vocab_file: Output, old_vocab_file: Output, new_vocab_offset: Long, num_new_vocab: Long, old_vocab_size: Long = -1L, name: String = "GenerateVocabRemapping") = run {
    buildOpTensors("GenerateVocabRemapping", name) {
      addInput(new_vocab_file, false)
      addInput(old_vocab_file, false)
      attr("new_vocab_offset", new_vocab_offset)
      attr("num_new_vocab", num_new_vocab)
      attr("old_vocab_size", old_vocab_size)
    }
  }
  
  fun _loadAndRemapMatrix(ckpt_path: Output, old_tensor_name: Output, row_remapping: Output, col_remapping: Output, initializing_values: Output, num_rows: Long, num_cols: Long, max_rows_in_memory: Long = -1L, name: String = "LoadAndRemapMatrix") = run {
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
}