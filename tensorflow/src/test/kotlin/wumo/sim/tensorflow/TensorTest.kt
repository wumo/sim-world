package wumo.sim.tensorflow

import org.junit.Test
import wumo.sim.tensorflow.tensor.Tensor
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.Shape
import wumo.sim.util.f
import wumo.sim.util.i
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.types.NDFloat

class TensorTest {
  @Test
  fun test() {
    val n = NDArray(Shape(2, 2), i(1, 2, 3, 4))
    val t = Tensor.fromNDArray(n, FLOAT)
    println(t)
  }
  
  @Test
  fun `get set`() {
    tf
    val _t = Tensor(Shape(2, 2), f(1f, 2f, 3f, 4f))
    val t = NDArray(Shape(2, 2), _t, NDFloat)
    println(t)
    t[0, 0] = 3f
    println(t)
    val t_c = t.copy()
    t_c[0, 0] = 6f
    println(t)
    println(t_c)
    val t2 = Tensor<Float>(_t.c_tensor)
    println(t2)
    val t3 = Tensor(1)
    println(t3)
    val a = tf.const(t3)
    tf.printGraph()
    tf.session {
      a.eval()
    }
  }
}