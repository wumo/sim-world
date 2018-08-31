/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output

interface gen_audio_ops {
  fun audioSpectrogram(input: Output, windowSize: Long, stride: Long, magnitudeSquared: Boolean = false, name: String = "AudioSpectrogram") = run {
    buildOpTensor("AudioSpectrogram", name) {
      addInput(input, false)
      attr("window_size", windowSize)
      attr("stride", stride)
      attr("magnitude_squared", magnitudeSquared)
    }
  }
  
  fun decodeWav(contents: Output, desiredChannels: Long = -1L, desiredSamples: Long = -1L, name: String = "DecodeWav") = run {
    buildOpTensors("DecodeWav", name) {
      addInput(contents, false)
      attr("desired_channels", desiredChannels)
      attr("desired_samples", desiredSamples)
    }
  }
  
  fun encodeWav(audio: Output, sampleRate: Output, name: String = "EncodeWav") = run {
    buildOpTensor("EncodeWav", name) {
      addInput(audio, false)
      addInput(sampleRate, false)
    }
  }
  
  fun mfcc(spectrogram: Output, sampleRate: Output, upperFrequencyLimit: Float = 4000.0f, lowerFrequencyLimit: Float = 20.0f, filterbankChannelCount: Long = 40L, dctCoefficientCount: Long = 13L, name: String = "Mfcc") = run {
    buildOpTensor("Mfcc", name) {
      addInput(spectrogram, false)
      addInput(sampleRate, false)
      attr("upper_frequency_limit", upperFrequencyLimit)
      attr("lower_frequency_limit", lowerFrequencyLimit)
      attr("filterbank_channel_count", filterbankChannelCount)
      attr("dct_coefficient_count", dctCoefficientCount)
    }
  }
}