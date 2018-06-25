package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow.TF_Operation

class Operation(graph: Graph, val nativeOp: TF_Operation) {
}