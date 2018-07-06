package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.helpers.println

class Session(val c_graph: TF_Graph) : AutoCloseable {
  private val c_session: TF_Session
  
  init {
    val status = TF_NewStatus()
    val opts = TF_NewSessionOptions()
    c_session = TF_NewSession(c_graph, opts, status)
    TF_DeleteSessionOptions(opts)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
  }
  
  override fun close() {
    val status = TF_NewStatus()
    TF_CloseSession(c_session, status)
    TF_DeleteSession(c_session, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
  }
  
  fun Operation.run() {
    val status = TF_NewStatus()
    TF_SessionRun(c_session, null, null, null, 0,
                  null, null, 0,
                  c_op, 1,
                  null, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
  }
  
  fun Tensor.eval() = op.name.println(":\n${eval<Any>()}")
  fun <T> Tensor.eval(): TensorValue<T> {
    val status = TF_NewStatus()
    val outputs = TF_Output(1)
    outputs.oper(op.c_op)
    outputs.index(value_index)
    val outputValues = TF_Tensor()
    TF_SessionRun(c_session, null, null, null, 0,
                  outputs, outputValues, 1,
                  null, 0,
                  null, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
    return TensorValue.wrap(outputValues)
  }
}