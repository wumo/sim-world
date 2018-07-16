package wumo.sim.algorithm.tensorflow

import org.junit.Test
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.util.f
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.x

class TensorBufferTest {
  @Test
  fun `get set`() {
    tf
    val _t = TensorBuffer(2 x 2, f(1f, 2f, 3f, 4f))
    val t = NDArray(2 x 2, _t)
    println(t)
    t[0, 0] = 3f
    println(t)
    val t2 = TensorBuffer<Float>(_t.c_tensor)
    println(t2)
    val t3 = TensorBuffer(1)
    println(t3)
    val a = tf.const(t3)
    tf.printGraph()
    tf.session {
      a.eval()
    }
  }
}