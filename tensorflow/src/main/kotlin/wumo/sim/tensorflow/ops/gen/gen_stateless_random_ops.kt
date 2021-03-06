/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT64

object gen_stateless_random_ops {
  fun statelessMultinomial(logits: Output, numSamples: Output, seed: Output, outputDtype: DataType<*> = INT64, name: String = "StatelessMultinomial"): Output =
      buildOpTensor("StatelessMultinomial", name) {
        addInput(logits, false)
        addInput(numSamples, false)
        addInput(seed, false)
        attr("output_dtype", outputDtype)
      }
  
  fun statelessRandomNormal(shape: Output, seed: Output, dtype: DataType<*> = FLOAT, name: String = "StatelessRandomNormal"): Output =
      buildOpTensor("StatelessRandomNormal", name) {
        addInput(shape, false)
        addInput(seed, false)
        attr("dtype", dtype)
      }
  
  fun statelessRandomUniform(shape: Output, seed: Output, dtype: DataType<*> = FLOAT, name: String = "StatelessRandomUniform"): Output =
      buildOpTensor("StatelessRandomUniform", name) {
        addInput(shape, false)
        addInput(seed, false)
        attr("dtype", dtype)
      }
  
  fun statelessTruncatedNormal(shape: Output, seed: Output, dtype: DataType<*> = FLOAT, name: String = "StatelessTruncatedNormal"): Output =
      buildOpTensor("StatelessTruncatedNormal", name) {
        addInput(shape, false)
        addInput(seed, false)
        attr("dtype", dtype)
      }
}