package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.tuples.tuple2
import wumo.sim.algorithm.util.tuples.tuple3
import wumo.sim.algorithm.util.tuples.tuple4
import wumo.sim.algorithm.util.tuples.tuple5

class SessionHelper(val nativeSession: Session) {
  val feeds = mutableListOf<Pair<Output, Tensor>>()
  val targets = mutableListOf<Operation>()
  
  fun feed(vararg feed: Pair<Output, Tensor>): SessionHelper {
    feeds += feed
    return this
  }
  
  fun target(target: Operation): SessionHelper {
    targets += target
    return this
  }
  
  fun accumulatedFeeds(): StringTensorPairVector {
    val ops = Array(feeds.size) { feeds[it].first.name().string }
    val feed_values = Array(feeds.size) { feeds[it].second }
    return StringTensorPairVector(ops, feed_values)
  }
  
  fun accumulatedTargets(): StringVector {
    val ops = Array(targets.size) { targets[it].node().name().string }
    return StringVector(*ops)
  }
  
  fun clear() {
    feeds.clear()
    targets.clear()
  }
  
  fun run(eval: StringVector = StringVector()): TensorVector {
    val outputs = TensorVector()
    nativeSession.Run(accumulatedFeeds(), eval, accumulatedTargets(), outputs)
    clear()
    return outputs
  }
  
  fun Operation.run(vararg feed: Pair<Output, Tensor>) {
    feeds += feed
    targets += this
    this@SessionHelper.run()
  }
  
  fun Operation.run() {
    targets += this
    this@SessionHelper.run()
  }
  
  fun Output.eval() {
    val outputs = run(StringVector(name().string))
    print("${name().string} = ")
    outputs.forEach { t ->
      println(t.DebugString().string)
    }
  }
  
  inline fun <reified T> eval(op: Output): WrappedTensor<T> {
    val outputs = run(StringVector(op.name().string))
    return TensorHelper.wrap(outputs[0])
  }
  
  inline fun <reified T1, reified T2> eval(op1: Output, op2: Output):
      tuple2<WrappedTensor<T1>, WrappedTensor<T2>> {
    val outputs = run(StringVector(op1.name().string, op2.name().string))
    return tuple2(TensorHelper.wrap(outputs[0]), TensorHelper.wrap(outputs[1]))
  }
  
  inline fun <reified T1, reified T2, reified T3> eval(op1: Output, op2: Output, op3: Output):
      tuple3<WrappedTensor<T1>, WrappedTensor<T2>, WrappedTensor<T3>> {
    val outputs = run(StringVector(op1.name().string, op2.name().string, op3.name().string))
    return tuple3(TensorHelper.wrap(outputs[0]), TensorHelper.wrap(outputs[1]), TensorHelper.wrap(outputs[2]))
  }
  
  inline fun <reified T1, reified T2, reified T3, reified T4> eval(op1: Output, op2: Output, op3: Output, op4: Output):
      tuple4<WrappedTensor<T1>, WrappedTensor<T2>, WrappedTensor<T3>, WrappedTensor<T4>> {
    val outputs = run(StringVector(op1.name().string, op2.name().string, op3.name().string, op4.name().string))
    return tuple4(TensorHelper.wrap(outputs[0]), TensorHelper.wrap(outputs[1]),
        TensorHelper.wrap(outputs[2]), TensorHelper.wrap(outputs[3]))
  }
  
  inline fun <reified T1, reified T2, reified T3, reified T4, reified T5>
      eval(op1: Output, op2: Output, op3: Output, op4: Output, op5: Output):
      tuple5<WrappedTensor<T1>, WrappedTensor<T2>, WrappedTensor<T3>, WrappedTensor<T4>, WrappedTensor<T5>> {
    val outputs = run(StringVector(op1.name().string, op2.name().string, op3.name().string, op4.name().string, op5.name().string))
    return tuple5(TensorHelper.wrap(outputs[0]), TensorHelper.wrap(outputs[1]),
        TensorHelper.wrap(outputs[2]), TensorHelper.wrap(outputs[3]), TensorHelper.wrap(outputs[4]))
  }
  
  fun eval(vararg ops: Output): Array<WrappedTensor<Any>> {
    val outputs = run(StringVector(*Array(ops.size) { ops[it].name().string }))
    return Array(outputs.size().toInt()) { TensorHelper.wrap<Any>(outputs[it.toLong()]) }
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