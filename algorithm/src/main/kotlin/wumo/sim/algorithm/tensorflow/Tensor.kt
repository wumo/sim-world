package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.tensorflow.*

/**
 * Represents one of the outputs of an `Operation`.
 *
 * `Tensor` is a symbolic handle to one of the outputs of an
 * `Operation`. It does not hold the values of that operation's output,
 * but instead provides a means of computing those values in a
 * TensorFlow [Session].
 *
 * This class has two primary purposes:
 * 1. A `Tensor` can be passed as an input to another `Operation`.
 * This builds a dataflow connection between operations, which
 * enables TensorFlow to execute an entire `Graph` that represents a
 * large, multi-step computation.
 *
 * 2. After the graph has been launched in a session, the value of the
 * `Tensor` can be computed by passing it to@{tf.Session.run}.
 * `t.eval()` is a shortcut for calling`tf.get_default_session().run(t)`.
 */
class Tensor(val op: Operation, val value_index: Int, val dtype: Int) {
  
  fun asTF_Output() = TF_Output().oper(op.c_op).index(value_index)
}