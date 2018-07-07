package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Session.newSession
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_SessionOptions.newSessionOptions
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.tuples.tuple2
import wumo.sim.algorithm.util.tuples.tuple3

class Session(val c_graph: TF_Graph) {
  private val c_session: TF_Session
  
  init {
    val status = newStatus()
    c_session = newSession(c_graph, newSessionOptions(), newStatus())
    throwExceptionIfNotOk(status)
  }
  
  val feed_dict = mutableListOf<Pair<Tensor, TensorValue<*>>>()
  val run_list = mutableListOf<Operation>()
  
  fun Operation.run(vararg feeds: Pair<Tensor, TensorValue<*>>) {
    feed_dict += feeds
    run_list += this
    eval()
  }
  
  fun clear() {
    feed_dict.clear()
    run_list.clear()
  }
  
  fun Array<Tensor>.eval() {
    val ts = eval(*this)
    for (i in 0 until size)
      this[i].print(ts[i])
  }
  
  fun eval(vararg fetch: Tensor): Array<TensorValue<Any>> {
    val status = newStatus()
    val (inputs, input_values, ninputs) = accumulateFeedDict()
    val (target_opers, ntargets) = accumulateRuns()
    val noutputs = fetch.size
    val outputs = TF_Output(noutputs.toLong())
    val output_values = PointerPointer<TF_Tensor>(noutputs.toLong())
    for ((i, f) in fetch.withIndex())
      outputs.position(i.toLong()).oper(f.op.c_op).index(f.value_index)
    outputs.position(0L)
    TF_SessionRun(c_session, null, inputs, input_values, ninputs,
                  outputs, output_values, noutputs,
                  target_opers, ntargets,
                  null, status)
    clear()
    return Array(noutputs) {
      TensorValue.wrap<Any>(output_values.get(TF_Tensor::class.java, it.toLong()))
    }
  }
  
  private fun accumulateFeedDict(): tuple3<TF_Output, PointerPointer<TF_Tensor>, Int> {
    val ninputs = feed_dict.size.toLong()
    val inputs = TF_Output(ninputs)
    val input_values = PointerPointer<TF_Tensor>(ninputs)
    for ((i, pair) in feed_dict.withIndex()) {
      val (input, input_value) = pair
      inputs.position(i.toLong()).oper(input.op.c_op).index(input.value_index)
      input_values.position(i.toLong()).put(input_value.c_tensor)
    }
    inputs.position(0L)
    input_values.position(0L)
    return tuple3(inputs, input_values, ninputs.toInt())
  }
  
  private fun accumulateRuns(): tuple2<PointerPointer<TF_Operation>, Int> {
    val ntargets = run_list.size.toLong()
    val target_opers = PointerPointer<TF_Operation>(ntargets)
    for ((i, op) in run_list.withIndex())
      target_opers.position(i.toLong()).put(op.c_op)
    target_opers.position(0L)
    return tuple2(target_opers, ntargets.toInt())
  }
  
  fun Tensor.eval() {
    print(eval<Any>())
  }
  
  fun <T> Tensor.eval(): TensorValue<T> {
    val t = eval(this)[0]
    return t as TensorValue<T>
  }
  
  private fun Tensor.print(v: TensorValue<*>) {
    val prefix = "${op.name}:${dtype.name()}$shape\n  ="
    println("$prefix${v.toString(3)}\n")
  }
}