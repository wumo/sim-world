package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.tuples.tuple2
import wumo.sim.algorithm.util.tuples.tuple3
import wumo.sim.algorithm.util.tuples.tuple4
import wumo.sim.algorithm.util.tuples.tuple5

class SessionHelper(val nativeSession: Session) {
  val feeds = mutableListOf<Pair<Output, Tensor>>()
  val targets = mutableListOf<Operation>()
  
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
    print("${name().string} = ")
    outputs.forEach { t ->
      println(t.DebugString().string)
    }
  }
  
  inline fun <reified T> eval(op: Output): WrappedTensor<T> {
    val outputs = TensorVector()
    nativeSession.Run(StringTensorPairVector(), StringVector(op.name().string), StringVector(), outputs)
    return TensorHelper.wrap(outputs[0])
  }
  
  inline fun <reified T1, reified T2> eval(op1: Output, op2: Output):
      tuple2<WrappedTensor<T1>, WrappedTensor<T2>> {
    val outputs = TensorVector()
    nativeSession.Run(StringTensorPairVector(),
        StringVector(op1.name().string, op2.name().string), StringVector(), outputs)
    return tuple2(TensorHelper.wrap(outputs[0]), TensorHelper.wrap(outputs[1]))
  }
  
  inline fun <reified T1, reified T2, reified T3> eval(op1: Output, op2: Output, op3: Output):
      tuple3<WrappedTensor<T1>, WrappedTensor<T2>, WrappedTensor<T3>> {
    val outputs = TensorVector()
    nativeSession.Run(StringTensorPairVector(),
        StringVector(op1.name().string, op2.name().string, op3.name().string), StringVector(), outputs)
    return tuple3(TensorHelper.wrap(outputs[0]), TensorHelper.wrap(outputs[1]), TensorHelper.wrap(outputs[2]))
  }
  
  inline fun <reified T1, reified T2, reified T3, reified T4> eval(op1: Output, op2: Output, op3: Output, op4: Output):
      tuple4<WrappedTensor<T1>, WrappedTensor<T2>, WrappedTensor<T3>, WrappedTensor<T4>> {
    val outputs = TensorVector()
    nativeSession.Run(StringTensorPairVector(),
        StringVector(op1.name().string, op2.name().string, op3.name().string, op4.name().string), StringVector(), outputs)
    return tuple4(TensorHelper.wrap(outputs[0]), TensorHelper.wrap(outputs[1]),
        TensorHelper.wrap(outputs[2]), TensorHelper.wrap(outputs[3]))
  }
  
  inline fun <reified T1, reified T2, reified T3, reified T4, reified T5>
      eval(op1: Output, op2: Output, op3: Output, op4: Output, op5: Output):
      tuple5<WrappedTensor<T1>, WrappedTensor<T2>, WrappedTensor<T3>, WrappedTensor<T4>, WrappedTensor<T5>> {
    val outputs = TensorVector()
    nativeSession.Run(StringTensorPairVector(),
        StringVector(op1.name().string, op2.name().string, op3.name().string, op4.name().string, op5.name().string),
        StringVector(), outputs)
    return tuple5(TensorHelper.wrap(outputs[0]), TensorHelper.wrap(outputs[1]),
        TensorHelper.wrap(outputs[2]), TensorHelper.wrap(outputs[3]), TensorHelper.wrap(outputs[4]))
  }
  
  fun eval(vararg ops: Output): Array<WrappedTensor<Any>> {
    val outputs = TensorVector()
    nativeSession.Run(StringTensorPairVector(),
        StringVector(*Array(ops.size) { ops[it].name().string }), StringVector(), outputs)
    
    return Array(outputs.size().toInt()) { TensorHelper.wrap<Any>(outputs[it.toLong()]) }
  }
  
  fun append(feed: Pair<Output, Tensor>): SessionHelper {
    feeds += feed
    return this
  }
  
  fun append(target: Operation): SessionHelper {
    targets += target
    return this
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