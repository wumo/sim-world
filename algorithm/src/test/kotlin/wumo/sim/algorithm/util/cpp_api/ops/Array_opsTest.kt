package wumo.sim.algorithm.util.cpp_api.ops

import org.junit.Test
import wumo.sim.algorithm.util.cpp_api.BaseTest
import wumo.sim.algorithm.util.x

class Array_opsTest : BaseTest() {
  @Test
  fun slice() {
    var i = 0f
    val input = tf.const(2 x 3 x 4, FloatArray(2 * 3 * 4) { i++ })
    val begin = tf.const(intArrayOf(1, 1, 2))
    val size = tf.const(intArrayOf(1, 2, 1))
    val slice = tf.slice(input, begin, size)
    println(tf.debugString())
    tf.session {
      slice.eval()
      val _s = eval<Float>(input)
      println(_s[1, 1, 2])
      println(_s[1, 2, 2])
    }
  }
  
  @Test
  fun onesLike() {
    val x = tf.const(2 x 3 x 4, 6f)
    val y = tf.onesLike(x)
    println(tf.debugString())
    tf.session {
      y.eval()
    }
  }
}