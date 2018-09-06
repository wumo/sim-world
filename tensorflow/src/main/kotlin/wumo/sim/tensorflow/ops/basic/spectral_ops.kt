package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_spectral_ops

object spectral_ops {
  interface API {
    fun batchFFT(input: Output, name: String = "BatchFFT"): Output {
      return gen_spectral_ops.batchFFT(input, name)
    }
    
    fun batchFFT2D(input: Output, name: String = "BatchFFT2D"): Output {
      return gen_spectral_ops.batchFFT2D(input, name)
    }
    
    fun batchFFT3D(input: Output, name: String = "BatchFFT3D"): Output {
      return gen_spectral_ops.batchFFT3D(input, name)
    }
    
    fun batchIFFT(input: Output, name: String = "BatchIFFT"): Output {
      return gen_spectral_ops.batchIFFT(input, name)
    }
    
    fun batchIFFT2D(input: Output, name: String = "BatchIFFT2D"): Output {
      return gen_spectral_ops.batchIFFT2D(input, name)
    }
    
    fun batchIFFT3D(input: Output, name: String = "BatchIFFT3D"): Output {
      return gen_spectral_ops.batchIFFT3D(input, name)
    }
    
    fun fFT(input: Output, name: String = "FFT"): Output {
      return gen_spectral_ops.fFT(input, name)
    }
    
    fun fFT2D(input: Output, name: String = "FFT2D"): Output {
      return gen_spectral_ops.fFT2D(input, name)
    }
    
    fun fFT3D(input: Output, name: String = "FFT3D"): Output {
      return gen_spectral_ops.fFT3D(input, name)
    }
    
    fun iFFT(input: Output, name: String = "IFFT"): Output {
      return gen_spectral_ops.iFFT(input, name)
    }
    
    fun iFFT2D(input: Output, name: String = "IFFT2D"): Output {
      return gen_spectral_ops.iFFT2D(input, name)
    }
    
    fun iFFT3D(input: Output, name: String = "IFFT3D"): Output {
      return gen_spectral_ops.iFFT3D(input, name)
    }
    
    fun iRFFT(input: Output, fftLength: Output, name: String = "IRFFT"): Output {
      return gen_spectral_ops.iRFFT(input, fftLength, name)
    }
    
    fun iRFFT2D(input: Output, fftLength: Output, name: String = "IRFFT2D"): Output {
      return gen_spectral_ops.iRFFT2D(input, fftLength, name)
    }
    
    fun iRFFT3D(input: Output, fftLength: Output, name: String = "IRFFT3D"): Output {
      return gen_spectral_ops.iRFFT3D(input, fftLength, name)
    }
    
    fun rFFT(input: Output, fftLength: Output, name: String = "RFFT"): Output {
      return gen_spectral_ops.rFFT(input, fftLength, name)
    }
    
    fun rFFT2D(input: Output, fftLength: Output, name: String = "RFFT2D"): Output {
      return gen_spectral_ops.rFFT2D(input, fftLength, name)
    }
    
    fun rFFT3D(input: Output, fftLength: Output, name: String = "RFFT3D"): Output {
      return gen_spectral_ops.rFFT3D(input, fftLength, name)
    }
  }
}