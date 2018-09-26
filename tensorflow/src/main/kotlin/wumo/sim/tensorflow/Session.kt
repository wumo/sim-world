@file:Suppress("UNCHECKED_CAST")

package wumo.sim.tensorflow

import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Session.newSession
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_SessionOptions.newSessionOptions
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.core.check
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputConvertible
import wumo.sim.tensorflow.tensor.Tensor
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.t2
import wumo.sim.util.t3
import wumo.sim.util.t4
import wumo.sim.util.t5

class Session(val c_graph: TF_Graph) {
  private val c_session: TF_Session
  
  init {
    val status = newStatus()
    c_session = newSession(c_graph, newSessionOptions(), newStatus())
    status.check()
  }
  
  val feed_dict = mutableListOf<Pair<Output, NDArray<*>>>()
  val run_list = mutableListOf<Op>()
  
  fun <T : Any> eval(t: OutputConvertible): NDArray<T> {
    val (t) = eval(listOf(t))
    return t as NDArray<T>
  }
  
  fun <T1 : Any, T2 : Any> eval(t1: OutputConvertible, t2: OutputConvertible): t2<NDArray<T1>, NDArray<T2>> {
    val (r1, r2) = eval(listOf(t1, t2))
    return t2(r1 as NDArray<T1>, r2 as NDArray<T2>)
  }
  
  fun <T1 : Any, T2 : Any, T3 : Any> eval(t1: OutputConvertible, t2: OutputConvertible, t3: OutputConvertible): t3<NDArray<T1>, NDArray<T2>, NDArray<T3>> {
    val (r1, r2, r3) = eval(listOf(t1, t2, t3))
    return t3(r1 as NDArray<T1>, r2 as NDArray<T2>,
              r3 as NDArray<T3>)
  }
  
  fun <T1 : Any, T2 : Any, T3 : Any, T4 : Any> eval(t1: OutputConvertible, t2: OutputConvertible, t3: OutputConvertible, t4: OutputConvertible):
      t4<NDArray<T1>, NDArray<T2>, NDArray<T3>, NDArray<T4>> {
    val (r1, r2, r3, r4) = eval(listOf(t1, t2, t3, t4))
    return t4(r1 as NDArray<T1>, r2 as NDArray<T2>,
              r3 as NDArray<T3>, r4 as NDArray<T4>)
  }
  
  fun <T1 : Any, T2 : Any, T3 : Any, T4 : Any, T5 : Any> eval(t1: OutputConvertible, t2: OutputConvertible, t3: OutputConvertible, t4: OutputConvertible, t5: OutputConvertible):
      t5<NDArray<T1>, NDArray<T2>, NDArray<T3>, NDArray<T4>, NDArray<T5>> {
    val (r1, r2, r3, r4, r5) = eval(listOf(t1, t2, t3, t4, t5))
    return t5(r1 as NDArray<T1>, r2 as NDArray<T2>,
              r3 as NDArray<T3>, r4 as NDArray<T4>,
              r5 as NDArray<T5>)
  }
  
  fun Op.run(vararg feeds: Pair<Output, NDArray<*>>) {
    feed_dict += feeds
    run_list += this
    eval(listOf())
  }
  
  fun OutputConvertible.eval() {
    print(eval<Any>(this.toOutput()))
  }
  
  fun List<OutputConvertible>.eval() {
    val ts = eval(this)
    for (i in 0 until size)
      this[i].print(ts[i])
  }
  
  private fun OutputConvertible.print(v: NDArray<*>) {
    val output = toOutput()
    val prefix = "${output.op.name}:${output.dataType.name}${output.shape}\n  ="
    println("$prefix${v.toString(3)}\n")
  }
  
  fun runString(fetch: List<String>,
                updates: List<String>,
                feed_dict: Map<String, NDArray<*>>): List<NDArray<*>> =
      run(fetch.map { tf.currentGraph.getTensor(it) },
          updates.map { tf.currentGraph.getTensor(it).op },
          feed_dict.mapKeys { tf.currentGraph.getTensor(it.key) })
  
  fun run(fetch: List<Output>,
          updates: List<Any>,
          feed_dict: Map<Output, NDArray<*>>): List<NDArray<*>> {
    run_list += updates.map {
      when (it) {
        is Output -> it.op
        is Op -> it
        else -> throw Exception()
      }
    }
    feed_dict.forEach { k, v -> this.feed_dict += k to v }
    return eval(fetch)
  }
  
  fun feed(vararg feeds: Pair<Output, NDArray<*>>) {
    feed_dict += feeds
  }
  
  fun target(vararg target: Op) {
    run_list += target
  }
  
  fun clear() {
    feed_dict.clear()
    run_list.clear()
  }
  
  fun eval(fetch: Iterable<OutputConvertible>): MutableList<NDArray<Any>> {
    val fetches = fetch.map { it.toOutput() }
    val status = newStatus()
    val (inputs, input_values, ninputs,tmp_tensors) = accumulateFeedDict()
    val (target_opers, ntargets) = accumulateRuns()
    val noutputs = fetches.size
    val outputs = TF_Output(noutputs.toLong())
    val output_values = PointerPointer<TF_Tensor>(noutputs.toLong())
    for ((i, f) in fetches.withIndex())
      outputs.position(i.toLong()).oper(f.op.c_op).index(f.valueIndex)
    outputs.position(0L)
    TF_SessionRun(c_session, null, inputs, input_values, ninputs,
                  outputs, output_values, noutputs,
                  target_opers, ntargets,
                  null, status)
    tmp_tensors
    status.check()
    clear()
    return MutableList(noutputs) {
      Tensor.toNDArray<Any>(output_values.get(TF_Tensor::class.java, it.toLong()))
    }
  }
  
  private fun accumulateFeedDict(): t4<TF_Output, PointerPointer<TF_Tensor>, Int, MutableList<Tensor<*>>> {
    val ninputs = feed_dict.size.toLong()
    val inputs = TF_Output(ninputs)
    val input_values = PointerPointer<TF_Tensor>(ninputs)
    val tmp_tensors = mutableListOf<Tensor<*>>()
    for ((i, pair) in feed_dict.withIndex()) {
      val (input, input_value) = pair
      inputs.position(i.toLong()).oper(input.op.c_op).index(input.valueIndex)
      val tensor = Tensor.fromNDArray(input_value, input.dataType)
      tmp_tensors += tensor
      input_values.position(i.toLong()).put(tensor.c_tensor)
    }
    inputs.position(0L)
    input_values.position(0L)
    return t4(inputs, input_values, ninputs.toInt(), tmp_tensors)
  }
  
  private fun accumulateRuns(): t2<PointerPointer<TF_Operation>, Int> {
    val ntargets = run_list.size.toLong()
    val target_opers = PointerPointer<TF_Operation>(ntargets)
    for ((i, op) in run_list.withIndex())
      target_opers.position(i.toLong()).put(op.c_op)
    target_opers.position(0L)
    return t2(target_opers, ntargets.toInt())
  }
  
}