package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.junit.Test
import wumo.sim.algorithm.tensorflow.ops.gen.relu
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.scalarDimension

class Math_gradKtTest {
  @Test
  fun `test gradient`() {
    val x = tf.placeholder(scalarDimension, dtype = DT_FLOAT, name = "x")
    val y = tf.relu(x, name = "y")
    
    val gradient = tf.gradients(y, listOf(x))
    tf.printGraph()
    tf.session {
      feed(x to NDArray.toNDArray(1f))
      for (ndArray in eval(gradient)) {
        println(ndArray)
      }
      feed(x to NDArray.toNDArray(2f))
      for (ndArray in eval(gradient)) {
        println(ndArray)
      }
    }
  }
}