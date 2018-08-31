/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output

interface gen_ctc_ops {
  fun cTCBeamSearchDecoder(inputs: Output, sequence_length: Output, beam_width: Long, top_paths: Long, merge_repeated: Boolean = true, name: String = "CTCBeamSearchDecoder") = run {
    buildOpTensors("CTCBeamSearchDecoder", name) {
      addInput(inputs, false)
      addInput(sequence_length, false)
      attr("beam_width", beam_width)
      attr("top_paths", top_paths)
      attr("merge_repeated", merge_repeated)
    }
  }
  
  fun cTCGreedyDecoder(inputs: Output, sequence_length: Output, merge_repeated: Boolean = false, name: String = "CTCGreedyDecoder") = run {
    buildOpTensors("CTCGreedyDecoder", name) {
      addInput(inputs, false)
      addInput(sequence_length, false)
      attr("merge_repeated", merge_repeated)
    }
  }
  
  fun cTCLoss(inputs: Output, labels_indices: Output, labels_values: Output, sequence_length: Output, preprocess_collapse_repeated: Boolean = false, ctc_merge_repeated: Boolean = true, ignore_longer_outputs_than_inputs: Boolean = false, name: String = "CTCLoss") = run {
    buildOpTensors("CTCLoss", name) {
      addInput(inputs, false)
      addInput(labels_indices, false)
      addInput(labels_values, false)
      addInput(sequence_length, false)
      attr("preprocess_collapse_repeated", preprocess_collapse_repeated)
      attr("ctc_merge_repeated", ctc_merge_repeated)
      attr("ignore_longer_outputs_than_inputs", ignore_longer_outputs_than_inputs)
    }
  }
}