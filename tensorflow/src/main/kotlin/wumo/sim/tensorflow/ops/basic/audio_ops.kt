package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_audio_ops

object audio_ops {
  interface API {
    fun audioSpectrogram(input: Output, windowSize: Long, stride: Long, magnitudeSquared: Boolean = false, name: String = "AudioSpectrogram"): Output {
      return gen_audio_ops.audioSpectrogram(input, windowSize, stride, magnitudeSquared, name)
    }
    
    fun decodeWav(contents: Output, desiredChannels: Long = -1L, desiredSamples: Long = -1L, name: String = "DecodeWav"): List<Output> {
      return gen_audio_ops.decodeWav(contents, desiredChannels, desiredSamples, name)
    }
    
    fun encodeWav(audio: Output, sampleRate: Output, name: String = "EncodeWav"): Output {
      return gen_audio_ops.encodeWav(audio, sampleRate, name)
    }
    
    fun mfcc(spectrogram: Output, sampleRate: Output, upperFrequencyLimit: Float = 4000.0f, lowerFrequencyLimit: Float = 20.0f, filterbankChannelCount: Long = 40L, dctCoefficientCount: Long = 13L, name: String = "Mfcc"): Output {
      return gen_audio_ops.mfcc(spectrogram, sampleRate, upperFrequencyLimit, lowerFrequencyLimit, filterbankChannelCount, dctCoefficientCount, name)
    }
  }
}