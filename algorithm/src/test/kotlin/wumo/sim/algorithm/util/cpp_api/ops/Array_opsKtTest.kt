package wumo.sim.algorithm.util.cpp_api.ops

import org.bytedeco.javacpp.tensorflow
import org.junit.Test

import org.junit.Assert.*
import wumo.sim.algorithm.util.cpp_api.BaseTest
import wumo.sim.algorithm.util.cpp_api.contrib.one_hot_encoding
import wumo.sim.algorithm.util.helpers.println
import wumo.sim.algorithm.util.x

class Array_opsKtTest : BaseTest() {
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
  
  @Test
  fun `one hot`() {
    val indices = tf.const(intArrayOf(0, 2, -1, 1), name = "indices")
    val depth = tf.const(3, name = "depth")
    val on_value = tf.const(5, name = "on_value")
    val off_value = tf.const(0, name = "off_value")
    
    val onehot = tf.oneHot(indices, depth, on_value, off_value)
    println(tf.debugString())
    tf.session {
      val result = eval<Int>(onehot)
      println(result)
    }
  }
  
  @Test
  fun `one hot encoding`() {
    val indices = tf.const(intArrayOf(0, 2, -1, 1), name = "indices")
    val onehot = tf.one_hot_encoding(indices, 3, 5f, 0f)
    
    println(tf.debugString())
    tf.session {
      val result = eval<Float>(onehot)
      println(result)
    }
  }
}