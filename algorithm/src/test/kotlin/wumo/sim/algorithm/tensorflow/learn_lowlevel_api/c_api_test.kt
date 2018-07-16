package wumo.sim.algorithm.tensorflow.learn_lowlevel_api

import org.bytedeco.javacpp.tensorflow
import org.junit.Test
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.tensorflow.ops.matmul
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.util.helpers.f
import wumo.sim.algorithm.util.helpers.println
import wumo.sim.algorithm.util.x

class c_api_test {
  @Test
  fun `update edge`() {
    val A = tf.const(1 x 4, f(1f, 2f, 3f, 4f))
    val B = tf.const(4 x 1, f(1f, 2f, 3f, 4f))
    val C = tf.matmul(A, B)
    val node = tensorflow.Node(C.op.c_op)
    node.DebugString().string.println()
    tf.printGraph()
    
  }
}