package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_checkpoint_ops

object checkpoint_ops {
  interface API {
    fun generateVocabRemapping(newVocabFile: Output, oldVocabFile: Output, newVocabOffset: Long, numNewVocab: Long, oldVocabSize: Long = -1L, name: String = "GenerateVocabRemapping"): List<Output> {
      return gen_checkpoint_ops.generateVocabRemapping(newVocabFile, oldVocabFile, newVocabOffset, numNewVocab, oldVocabSize, name)
    }
    
    fun loadAndRemapMatrix(ckptPath: Output, oldTensorName: Output, rowRemapping: Output, colRemapping: Output, initializingValues: Output, numRows: Long, numCols: Long, maxRowsInMemory: Long = -1L, name: String = "LoadAndRemapMatrix"): Output {
      return gen_checkpoint_ops.loadAndRemapMatrix(ckptPath, oldTensorName, rowRemapping, colRemapping, initializingValues, numRows, numCols, maxRowsInMemory, name)
    }
  }
}