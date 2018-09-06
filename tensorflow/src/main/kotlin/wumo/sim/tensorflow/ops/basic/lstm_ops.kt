package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_lstm_ops

object lstm_ops {
  interface API {
    fun blockLSTM(seqLenMax: Output, x: Output, csPrev: Output, hPrev: Output, w: Output, wci: Output, wcf: Output, wco: Output, b: Output, forgetBias: Float = 1.0f, cellClip: Float = 3.0f, usePeephole: Boolean = false, name: String = "BlockLSTM"): List<Output> {
      return gen_lstm_ops.blockLSTM(seqLenMax, x, csPrev, hPrev, w, wci, wcf, wco, b, forgetBias, cellClip, usePeephole, name)
    }
    
    fun blockLSTMGrad(seqLenMax: Output, x: Output, csPrev: Output, hPrev: Output, w: Output, wci: Output, wcf: Output, wco: Output, b: Output, i: Output, cs: Output, f: Output, o: Output, ci: Output, co: Output, h: Output, csGrad: Output, hGrad: Output, usePeephole: Boolean, name: String = "BlockLSTMGrad"): List<Output> {
      return gen_lstm_ops.blockLSTMGrad(seqLenMax, x, csPrev, hPrev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, csGrad, hGrad, usePeephole, name)
    }
    
    fun lSTMBlockCell(x: Output, csPrev: Output, hPrev: Output, w: Output, wci: Output, wcf: Output, wco: Output, b: Output, forgetBias: Float = 1.0f, cellClip: Float = 3.0f, usePeephole: Boolean = false, name: String = "LSTMBlockCell"): List<Output> {
      return gen_lstm_ops.lSTMBlockCell(x, csPrev, hPrev, w, wci, wcf, wco, b, forgetBias, cellClip, usePeephole, name)
    }
    
    fun lSTMBlockCellGrad(x: Output, csPrev: Output, hPrev: Output, w: Output, wci: Output, wcf: Output, wco: Output, b: Output, i: Output, cs: Output, f: Output, o: Output, ci: Output, co: Output, csGrad: Output, hGrad: Output, usePeephole: Boolean, name: String = "LSTMBlockCellGrad"): List<Output> {
      return gen_lstm_ops.lSTMBlockCellGrad(x, csPrev, hPrev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, csGrad, hGrad, usePeephole, name)
    }
  }
}