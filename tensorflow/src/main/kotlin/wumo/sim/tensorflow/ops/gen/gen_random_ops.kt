/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.ops.Output
import wumo.sim.util.Shape
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.tf
import wumo.sim.util.ndarray.NDArray

interface gen_random_ops {
fun _multinomial(logits: Output, num_samples: Output, seed: Long = 0L, seed2: Long = 0L, output_dtype: Int = DT_INT64, name: String = "Multinomial") = run {
buildOpTensor("Multinomial", name){
addInput(logits,false)
addInput(num_samples,false)
attr("seed", seed)
attr("seed2", seed2)
attrType("output_dtype", output_dtype)
}
}
fun _parameterizedTruncatedNormal(shape: Output, means: Output, stdevs: Output, minvals: Output, maxvals: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "ParameterizedTruncatedNormal") = run {
buildOpTensor("ParameterizedTruncatedNormal", name){
addInput(shape,false)
addInput(means,false)
addInput(stdevs,false)
addInput(minvals,false)
addInput(maxvals,false)
attr("seed", seed)
attr("seed2", seed2)
}
}
fun _randomGamma(shape: Output, alpha: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomGamma") = run {
buildOpTensor("RandomGamma", name){
addInput(shape,false)
addInput(alpha,false)
attr("seed", seed)
attr("seed2", seed2)
}
}
fun _randomPoissonV2(shape: Output, rate: Output, seed: Long = 0L, seed2: Long = 0L, dtype: Int = DT_INT64, name: String = "RandomPoissonV2") = run {
buildOpTensor("RandomPoissonV2", name){
addInput(shape,false)
addInput(rate,false)
attr("seed", seed)
attr("seed2", seed2)
attrType("dtype", dtype)
}
}
fun _randomShuffle(value: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomShuffle") = run {
buildOpTensor("RandomShuffle", name){
addInput(value,false)
attr("seed", seed)
attr("seed2", seed2)
}
}
fun _randomStandardNormal(shape: Output, dtype: Int, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomStandardNormal") = run {
buildOpTensor("RandomStandardNormal", name){
addInput(shape,false)
attrType("dtype", dtype)
attr("seed", seed)
attr("seed2", seed2)
}
}
fun _randomUniform(shape: Output, dtype: Int, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomUniform") = run {
buildOpTensor("RandomUniform", name){
addInput(shape,false)
attrType("dtype", dtype)
attr("seed", seed)
attr("seed2", seed2)
}
}
fun _randomUniformInt(shape: Output, minval: Output, maxval: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomUniformInt") = run {
buildOpTensor("RandomUniformInt", name){
addInput(shape,false)
addInput(minval,false)
addInput(maxval,false)
attr("seed", seed)
attr("seed2", seed2)
}
}
fun _truncatedNormal(shape: Output, dtype: Int, seed: Long = 0L, seed2: Long = 0L, name: String = "TruncatedNormal") = run {
buildOpTensor("TruncatedNormal", name){
addInput(shape,false)
attrType("dtype", dtype)
attr("seed", seed)
attr("seed2", seed2)
}
}
fun _randomGammaGrad(alpha: Output, sample: Output, name: String = "RandomGammaGrad") = run {
buildOpTensor("RandomGammaGrad", name){
addInput(alpha,false)
addInput(sample,false)
}
}
}