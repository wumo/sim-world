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

interface gen_audio_ops {
fun _audioSpectrogram(input: Output, window_size: Long, stride: Long, magnitude_squared: Boolean = false, name: String = "AudioSpectrogram") = run {
buildOpTensor("AudioSpectrogram", name){
addInput(input,false)
attr("window_size", window_size)
attr("stride", stride)
attr("magnitude_squared", magnitude_squared)
}
}
fun _decodeWav(contents: Output, desired_channels: Long = -1L, desired_samples: Long = -1L, name: String = "DecodeWav") = run {
buildOpTensors("DecodeWav", name){
addInput(contents,false)
attr("desired_channels", desired_channels)
attr("desired_samples", desired_samples)
}
}
fun _encodeWav(audio: Output, sample_rate: Output, name: String = "EncodeWav") = run {
buildOpTensor("EncodeWav", name){
addInput(audio,false)
addInput(sample_rate,false)
}
}
fun _mfcc(spectrogram: Output, sample_rate: Output, upper_frequency_limit: Float = 4000.0f, lower_frequency_limit: Float = 20.0f, filterbank_channel_count: Long = 40L, dct_coefficient_count: Long = 13L, name: String = "Mfcc") = run {
buildOpTensor("Mfcc", name){
addInput(spectrogram,false)
addInput(sample_rate,false)
attr("upper_frequency_limit", upper_frequency_limit)
attr("lower_frequency_limit", lower_frequency_limit)
attr("filterbank_channel_count", filterbank_channel_count)
attr("dct_coefficient_count", dct_coefficient_count)
}
}
}