package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.tensorflow.Tensor.create
import wumo.sim.algorithm.util.Dimension

fun tensor(shape: Dimension, data: FloatArray) =
    create(data, MakeShape(shape.asLongArray()))