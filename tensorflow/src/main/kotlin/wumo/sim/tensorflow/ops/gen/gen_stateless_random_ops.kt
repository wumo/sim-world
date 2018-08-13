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

interface gen_stateless_random_ops {
  fun _statelessMultinomial(logits: Output, num_samples: Output, seed: Output, output_dtype: Int = DT_INT64, name: String = "StatelessMultinomial") = run {
    buildOpTensor("StatelessMultinomial", name) {
      addInput(logits, false)
      addInput(num_samples, false)
      addInput(seed, false)
      attrType("output_dtype", output_dtype)
    }
  }
  
  fun _statelessRandomNormal(shape: Output, seed: Output, dtype: Int = DT_FLOAT, name: String = "StatelessRandomNormal") = run {
    buildOpTensor("StatelessRandomNormal", name) {
      addInput(shape, false)
      addInput(seed, false)
      attrType("dtype", dtype)
    }
  }
  
  fun _statelessRandomUniform(shape: Output, seed: Output, dtype: Int = DT_FLOAT, name: String = "StatelessRandomUniform") = run {
    buildOpTensor("StatelessRandomUniform", name) {
      addInput(shape, false)
      addInput(seed, false)
      attrType("dtype", dtype)
    }
  }
  
  fun _statelessTruncatedNormal(shape: Output, seed: Output, dtype: Int = DT_FLOAT, name: String = "StatelessTruncatedNormal") = run {
    buildOpTensor("StatelessTruncatedNormal", name) {
      addInput(shape, false)
      addInput(seed, false)
      attrType("dtype", dtype)
    }
  }
}