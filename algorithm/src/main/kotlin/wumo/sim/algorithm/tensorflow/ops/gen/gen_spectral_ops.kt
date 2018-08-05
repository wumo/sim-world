/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.algorithm.tensorflow.ops.gen

import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.buildOpTensor
import wumo.sim.algorithm.tensorflow.tf

object gen_spectral_ops {
  fun fFT(input: Tensor, name: String = "FFT") = run {
    tf.buildOpTensor("FFT", name) {
      addInput(input, false)
      
    }
  }
  
  fun fFT2D(input: Tensor, name: String = "FFT2D") = run {
    tf.buildOpTensor("FFT2D", name) {
      addInput(input, false)
      
    }
  }
  
  fun fFT3D(input: Tensor, name: String = "FFT3D") = run {
    tf.buildOpTensor("FFT3D", name) {
      addInput(input, false)
      
    }
  }
  
  fun iFFT(input: Tensor, name: String = "IFFT") = run {
    tf.buildOpTensor("IFFT", name) {
      addInput(input, false)
      
    }
  }
  
  fun iFFT2D(input: Tensor, name: String = "IFFT2D") = run {
    tf.buildOpTensor("IFFT2D", name) {
      addInput(input, false)
      
    }
  }
  
  fun iFFT3D(input: Tensor, name: String = "IFFT3D") = run {
    tf.buildOpTensor("IFFT3D", name) {
      addInput(input, false)
      
    }
  }
  
  fun iRFFT(input: Tensor, fft_length: Tensor, name: String = "IRFFT") = run {
    tf.buildOpTensor("IRFFT", name) {
      addInput(input, false)
      addInput(fft_length, false)
      
    }
  }
  
  fun iRFFT2D(input: Tensor, fft_length: Tensor, name: String = "IRFFT2D") = run {
    tf.buildOpTensor("IRFFT2D", name) {
      addInput(input, false)
      addInput(fft_length, false)
      
    }
  }
  
  fun iRFFT3D(input: Tensor, fft_length: Tensor, name: String = "IRFFT3D") = run {
    tf.buildOpTensor("IRFFT3D", name) {
      addInput(input, false)
      addInput(fft_length, false)
      
    }
  }
  
  fun rFFT(input: Tensor, fft_length: Tensor, name: String = "RFFT") = run {
    tf.buildOpTensor("RFFT", name) {
      addInput(input, false)
      addInput(fft_length, false)
      
    }
  }
  
  fun rFFT2D(input: Tensor, fft_length: Tensor, name: String = "RFFT2D") = run {
    tf.buildOpTensor("RFFT2D", name) {
      addInput(input, false)
      addInput(fft_length, false)
      
    }
  }
  
  fun rFFT3D(input: Tensor, fft_length: Tensor, name: String = "RFFT3D") = run {
    tf.buildOpTensor("RFFT3D", name) {
      addInput(input, false)
      addInput(fft_length, false)
      
    }
  }
}