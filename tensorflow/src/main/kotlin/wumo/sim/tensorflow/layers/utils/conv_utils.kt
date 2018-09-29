package wumo.sim.tensorflow.layers.utils

import wumo.sim.tensorflow.contrib.layers.CNNDataFormat
import wumo.sim.tensorflow.contrib.layers.CNNDataFormat.*
import wumo.sim.tensorflow.layers.utils.DataFormat.channels_first
import wumo.sim.tensorflow.layers.utils.DataFormat.channels_last

enum class DataFormat(val str: String) {
  channels_first("channels_first"),
  channels_last("channels_last")
}

fun DataFormat.toCNNDataFormat(ndim: Int): CNNDataFormat =
    when (this) {
      channels_last -> when (ndim) {
        3 -> NWC
        4 -> NHWC
        5 -> NDHWC
        else -> error("Input rank not supported: $ndim")
      }
      channels_first -> when (ndim) {
        3 -> NCW
        4 -> NCHW
        5 -> NCDHW
        else -> error("Input rank not supported: $ndim")
      }
    }