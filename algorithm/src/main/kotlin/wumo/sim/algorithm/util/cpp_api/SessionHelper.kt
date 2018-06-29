package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.tensorflow.*

class SessionHelper(val nativeSession: Session) {
  fun Operation.run(feed: Pair<Output, Tensor>) {
    val (op, tensor) = feed
    nativeSession.Run(StringTensorPairVector(arrayOf(op.name().string), arrayOf(tensor)),
        StringVector(), StringVector(node().name().string), TensorVector())
  }
  
  fun Operation.run() {
    nativeSession.Run(StringTensorPairVector(), StringVector(),
        StringVector(this.node().name().string), TensorVector())
  }
  
  fun Output.eval() {
    val outputs = TensorVector()
    nativeSession.Run(StringTensorPairVector(), StringVector(name().string), StringVector(), outputs)
    outputs.forEach { t ->
      println(t.DebugString().string)
    }
  }
}

inline fun TensorVector.forEach(block: (Tensor) -> Unit) {
  val iter = begin()
  val end = end()
  while (!iter.equals(end)) {
    block(iter.get())
    iter.increment()
  }
}