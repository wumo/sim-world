@file:Suppress("UNCHECKED_CAST")

package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Session.newSession
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_SessionOptions.newSessionOptions
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.ops.Op
import wumo.sim.algorithm.tensorflow.ops.Output
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.tuple2
import wumo.sim.util.tuple3
import wumo.sim.util.tuple4
import wumo.sim.util.tuple5

class Session(val c_graph: TF_Graph) {
  private val c_session: TF_Session
  
  init {
    val status = newStatus()
    c_session = newSession(c_graph, newSessionOptions(), newStatus())
    throwExceptionIfNotOk(status)
  }
  
  val feed_dict = mutableListOf<Pair<Output, NDArray<*>>>()
  val run_list = mutableListOf<Op>()
  
  fun Op.run(vararg feeds: Pair<Output, NDArray<*>>) {
    feed_dict += feeds
    run_list += this
    _eval()
  }
  
  fun clear() {
    feed_dict.clear()
    run_list.clear()
  }
  
  fun Array<Output>.eval() {
    val ts = _eval(*this)
    for (i in 0 until size)
      this[i].print(ts[i])
  }
  
  fun eval() = _eval()
  
  fun <T : Any> eval(t: Output): NDArray<T> {
    val (t) = _eval(t)
    return t as NDArray<T>
  }
  
  fun <T1 : Any, T2 : Any> eval(t1: Output, t2: Output): tuple2<NDArray<T1>, NDArray<T2>> {
    val (r1, r2) = _eval(t1, t2)
    return tuple2(r1 as NDArray<T1>, r2 as NDArray<T2>)
  }
  
  fun <T1 : Any, T2 : Any, T3 : Any> eval(t1: Output, t2: Output, t3: Output): tuple3<NDArray<T1>, NDArray<T2>, NDArray<T3>> {
    val (r1, r2, r3) = _eval(t1, t2, t3)
    return tuple3(r1 as NDArray<T1>, r2 as NDArray<T2>,
                  r3 as NDArray<T3>)
  }
  
  fun <T1 : Any, T2 : Any, T3 : Any, T4 : Any> eval(t1: Output, t2: Output, t3: Output, t4: Output):
      tuple4<NDArray<T1>, NDArray<T2>, NDArray<T3>, NDArray<T4>> {
    val (r1, r2, r3, r4) = _eval(t1, t2, t3, t4)
    return tuple4(r1 as NDArray<T1>, r2 as NDArray<T2>,
                  r3 as NDArray<T3>, r4 as NDArray<T4>)
  }
  
  fun <T1 : Any, T2 : Any, T3 : Any, T4 : Any, T5 : Any> eval(t1: Output, t2: Output, t3: Output, t4: Output, t5: Output):
      tuple5<NDArray<T1>, NDArray<T2>, NDArray<T3>, NDArray<T4>, NDArray<T5>> {
    val (r1, r2, r3, r4, r5) = _eval(t1, t2, t3, t4, t5)
    return tuple5(r1 as NDArray<T1>, r2 as NDArray<T2>,
                  r3 as NDArray<T3>, r4 as NDArray<T4>,
                  r5 as NDArray<T5>)
  }
  
  fun eval(fetch: List<Output>): Array<NDArray<Any>> {
    val status = newStatus()
    val (inputs, input_values, ninputs) = accumulateFeedDict()
    val (target_opers, ntargets) = accumulateRuns()
    val noutputs = fetch.size
    val outputs = TF_Output(noutputs.toLong())
    val output_values = PointerPointer<TF_Tensor>(noutputs.toLong())
    for ((i, f) in fetch.withIndex())
      outputs.position(i.toLong()).oper(f.op!!.c_op).index(f.value_index)
    outputs.position(0L)
    TF_SessionRun(c_session, null, inputs, input_values, ninputs,
                  outputs, output_values, noutputs,
                  target_opers, ntargets,
                  null, status)
    throwExceptionIfNotOk(status)
    clear()
    return Array(noutputs) {
      Tensor.toNDArray<Any>(output_values.get(TF_Tensor::class.java, it.toLong()))
    }
  }
  
  fun _eval(vararg fetch: Output): Array<NDArray<Any>> {
    val status = newStatus()
    val (inputs, input_values, ninputs) = accumulateFeedDict()
    val (target_opers, ntargets) = accumulateRuns()
    val noutputs = fetch.size
    val outputs = TF_Output(noutputs.toLong())
    val output_values = PointerPointer<TF_Tensor>(noutputs.toLong())
    for ((i, f) in fetch.withIndex())
      outputs.position(i.toLong()).oper(f.op!!.c_op).index(f.value_index)
    outputs.position(0L)
    TF_SessionRun(c_session, null, inputs, input_values, ninputs,
                  outputs, output_values, noutputs,
                  target_opers, ntargets,
                  null, status)
    throwExceptionIfNotOk(status)
    clear()
    return Array(noutputs) {
      Tensor.toNDArray<Any>(output_values.get(TF_Tensor::class.java, it.toLong()))
    }
  }
  
  private fun accumulateFeedDict(): tuple3<TF_Output, PointerPointer<TF_Tensor>, Int> {
    val ninputs = feed_dict.size.toLong()
    val inputs = TF_Output(ninputs)
    val input_values = PointerPointer<TF_Tensor>(ninputs)
    for ((i, pair) in feed_dict.withIndex()) {
      val (input, input_value) = pair
      inputs.position(i.toLong()).oper(input.op!!.c_op).index(input.value_index)
      input_values.position(i.toLong()).put(Tensor.fromNDArray(input_value).c_tensor)
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
  
  fun Output.eval() {
    print(eval<Any>(this))
  }
  
  fun Op.eval() {
  
  }
  
  private fun Output.print(v: NDArray<*>) {
    val prefix = "${op!!.name}:${dtype.name()}$shape\n  ="
    println("$prefix${v.toString(3)}\n")
  }
  
  fun feed(vararg feeds: Pair<Output, NDArray<*>>) {
    feed_dict += feeds
  }
  
  fun target(vararg target: Op) {
    run_list += target
  }
  
  fun run(fetch: Array<String>,
          updates: Array<String>,
          feed_dict: Map<String, NDArray<*>>): Array<NDArray<*>> {
    val ninputs = feed_dict.size
    val inputs = TF_Output(ninputs.toLong())
    val input_values = PointerPointer<TF_Tensor>(ninputs.toLong())
    for ((i, pair) in feed_dict.entries.withIndex()) {
      val (_input, input_value) = pair
      val input = tf.g.getTensor(_input)
      inputs.position(i.toLong()).oper(input.op!!.c_op).index(input.value_index)
      input_values.position(i.toLong()).put(Tensor.fromNDArray(input_value).c_tensor)
    }
    inputs.position(0L)
    input_values.position(0L)
    
    val ntargets = updates.size
    val target_opers = PointerPointer<TF_Operation>(ntargets.toLong())
    for ((i, op) in updates.withIndex()) {
      val op = tf.g.getTensor(op).op!!
      target_opers.position(i.toLong()).put(op.c_op)
    }
    target_opers.position(0L)
    
    val status = newStatus()
    val noutputs = fetch.size
    val outputs = TF_Output(noutputs.toLong())
    val output_values = PointerPointer<TF_Tensor>(noutputs.toLong())
    for ((i, _f) in fetch.withIndex()) {
      val f = tf.g.getTensor(_f)
      outputs.position(i.toLong()).oper(f.op!!.c_op).index(f.value_index)
    }
    outputs.position(0L)
    TF_SessionRun(c_session, null, inputs, input_values, ninputs,
                  outputs, output_values, noutputs,
                  target_opers, ntargets,
                  null, status)
    throwExceptionIfNotOk(status)
    clear()
    return Array(noutputs) {
      Tensor.toNDArray<Any>(output_values.get(TF_Tensor::class.java, it.toLong()))
    }
  }
  
  fun run(fetch: Array<Output>,
          updates: Array<out Any>,
          feed_dict: Map<Output, NDArray<*>>): Array<NDArray<*>> {
    val ninputs = feed_dict.size
    val inputs = TF_Output(ninputs.toLong())
    val input_values = PointerPointer<TF_Tensor>(ninputs.toLong())
    for ((i, pair) in feed_dict.entries.withIndex()) {
      val (input, input_value) = pair
      inputs.position(i.toLong()).oper(input.op!!.c_op).index(input.value_index)
      input_values.position(i.toLong()).put(Tensor.fromNDArray(input_value).c_tensor)
    }
    inputs.position(0L)
    input_values.position(0L)
    
    val ntargets = updates.size
    val target_opers = PointerPointer<TF_Operation>(ntargets.toLong())
    for ((i, op) in updates.withIndex()) {
      val op = when (op) {
        is Output -> op.op
        is Op -> op
        else -> throw Exception()
      }
      target_opers.position(i.toLong()).put(op!!.c_op)
    }
    target_opers.position(0L)
    
    val status = newStatus()
    val noutputs = fetch.size
    val outputs = TF_Output(noutputs.toLong())
    val output_values = PointerPointer<TF_Tensor>(noutputs.toLong())
    for ((i, f) in fetch.withIndex())
      outputs.position(i.toLong()).oper(f.op!!.c_op).index(f.value_index)
    outputs.position(0L)
    TF_SessionRun(c_session, null, inputs, input_values, ninputs,
                  outputs, output_values, noutputs,
                  target_opers, ntargets,
                  null, status)
    throwExceptionIfNotOk(status)
    clear()
    return Array(noutputs) {
      Tensor.toNDArray<Any>(output_values.get(TF_Tensor::class.java, it.toLong()))
    }
  }
}