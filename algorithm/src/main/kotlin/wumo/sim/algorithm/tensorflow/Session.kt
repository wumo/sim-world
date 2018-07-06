package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.helper.tensorflow
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Session.newSession
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_SessionOptions.newSessionOptions
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.helpers.println

class Session(val c_graph: TF_Graph) {
  private val c_session: TF_Session
  
  init {
    val status = newStatus()
    c_session = newSession(c_graph, newSessionOptions(), newStatus())
    throwExceptionIfNotOk(status)
  }
  
  fun Operation.run() {
    val status = newStatus()
    TF_SessionRun(c_session, null, null, null, 0,
                  null, null, 0,
                  c_op, 1,
                  null, status)
    throwExceptionIfNotOk(status)
  }
  
  fun Tensor.eval() = op.name.println(":\n${eval<Any>()}")
  fun <T> Tensor.eval(): TensorValue<T> {
    val status = newStatus()
    val outputs = TF_Output(1)
    outputs.oper(op.c_op)
    outputs.index(value_index)
    val outputValues = TF_Tensor()
    TF_SessionRun(c_session, null, null, null, 0,
                  outputs, outputValues, 1,
                  null, 0,
                  null, status)
    
    throwExceptionIfNotOk(status)
    return TensorValue.wrap(outputValues)
  }
}