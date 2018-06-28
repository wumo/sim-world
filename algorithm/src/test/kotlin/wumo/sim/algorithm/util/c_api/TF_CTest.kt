package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.TF_Version
import org.junit.After
import org.junit.Assert
import org.junit.Before
import org.junit.Test
import org.tensorflow.framework.DataType
import org.tensorflow.framework.GraphDef
import wumo.sim.algorithm.util.c_api.core.const
import wumo.sim.algorithm.util.c_api.core.placeholder
import wumo.sim.algorithm.util.c_api.core.variable
import wumo.sim.algorithm.util.c_api.math_ops.*
import wumo.sim.algorithm.util.c_api.random_ops.random_uniform
import wumo.sim.algorithm.util.x
import java.nio.FloatBuffer

class TF_CTest {
  lateinit var tf: TF_C
  @Before
  fun setup() {
    tf = TF_C()
  }
  
  @After
  fun tearUp() {
  }
  
  @Test
  fun `print version`() {
    println(TF_Version().string)
  }
  
  @Test
  fun `attrValueProto test`() {
    val attrValue = tensorflow.AttrValue()
    attrValue.mutable_tensor().apply {
      set_dtype(tensorflow.DT_FLOAT)
      mutable_tensor_shape().apply {
        add_dim().set_size(16)
        add_dim().set_size(4)
      }
      add_float_val(9f)
    }
    attrValue.use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.DT_FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      val buf = FloatBuffer.allocate(64)
      A.writeTo(buf)
      buf.flip()
      println(buf.remaining())
      while (buf.hasRemaining())
        println(buf.get())
      println(buf.remaining())
    }
  }
  
  @Test
  fun `const def test`() {
    val A = tf.const(16 x 4, 9f)
    
    tf.session {
      A.eval()
    }
  }
  
  @Test
  fun `placeholder test`() {
    val A = tf.placeholder(2 x 2 x 2)
    val B = tf.variable(2 x 2 x 2, A)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      feedAndTarget(A, Tensor.create(arrayOf(
          arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f)),
          arrayOf(floatArrayOf(5f, 6f), floatArrayOf(7f, 8f)))),
          init)
      B.eval()
    }
  }
  
  @Test
  fun `variable def test`() {
    val A = tf.variable(16 x 4, 9f)
    val B = tf.variable(16 x 4, 9f)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      target(init)
      A.eval()
    }
  }
  
  @Test
  fun `random uniform test`() {
    val rand = tf.random_uniform(16 x 4)
    val A = tf.variable(16 x 4, rand)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      target(init)
      A.eval()
    }
  }
  
  @Test
  fun `random uniform test2`() {
    val rand = tf.random_uniform(16 x 4, 2f, 4f)
    val A = tf.variable(16 x 4, rand)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      target(init)
      A.eval()
    }
  }
  
  @Test
  fun `add sub mul div test`() {
    val A = tf.const(100f)
    val B = tf.const(2f)
    val `A+B` = tf.add(A, B)
    val `A-B` = tf.sub(A, B)
    val `A*B` = tf.mul(A, B)
    val `A div B` = tf.div(A, B)
    
    println(tf.debugString())
    tf.session {
      Assert.assertEquals(102f, `A+B`.fetch().floatValue())
      Assert.assertEquals(98f, `A-B`.fetch().floatValue())
      Assert.assertEquals(200f, `A*B`.fetch().floatValue())
      Assert.assertEquals(50f, `A div B`.fetch().floatValue())
    }
  }
  
  @Test
  fun `add sub mul div broadcast test`() {
    val A = tf.const(2 x 3, 100f)
    val B = tf.const(2f)
    val `A+B` = tf.add(A, B)
    val `A-B` = tf.sub(A, B)
    val `A*B` = tf.mul(A, B)
    val `A div B` = tf.div(A, B)
    
    println(tf.debugString())
    tf.session {
      `A+B`.eval()
      `A-B`.eval()
      `A*B`.eval()
      `A div B`.eval()
    }
  }
  
  @Test
  fun `sum test 1`() {
    val A = tf.const(floatArrayOf(1f, 2f, 3f), name = "A")
    val axis = tf.const(intArrayOf(0), name = "axis")
    val sum = tf.sum(A, axis, name = "sum")
    println(tf.debugString())
    tf.session {
      sum.eval()
    }
  }
  
  @Test
  fun `sum test 2`() {
    val A = tf.const(arrayOf(floatArrayOf(1f, 2f, 3f)), name = "A")
    val axis = tf.const(intArrayOf(0, 1), name = "axis")
    val sum = tf.sum(A, axis, name = "sum")
    println(tf.debugString())
    tf.session {
      sum.eval()
    }
  }
  
  @Test
  fun `matmul test`() {
    val A = tf.const(arrayOf(floatArrayOf(1f, 2f, 3f), floatArrayOf(4f, 5f, 6f)), "A")
    val B = tf.const(arrayOf(floatArrayOf(7f, 8f), floatArrayOf(9f, 10f), floatArrayOf(11f, 12f)), "B")
    val matmul = tf.matmul(A, B, "matmul")
    println(tf.debugString())
    tf.session {
      matmul.eval()
    }
  }
  
  @Test
  fun `square test`() {
    val A = tf.const(arrayOf(floatArrayOf(1f, 2f, 3f), floatArrayOf(4f, 5f, 6f)), "A")
    val square = tf.square(A, "square")
    println(tf.debugString())
    tf.session {
      square.eval()
    }
  }
  
  @Test
  fun `argmax test`() {
    val A = tf.const(arrayOf(floatArrayOf(3f, 2f, 3f), floatArrayOf(1f, 5f, 6f)), "A")
    val dim = tf.const(0)
    val argmax = tf.argmax(A, dim, "argmax")
    println(tf.debugString())
    tf.session {
      argmax.eval()
    }
  }
  
  @Test
  fun `linear layer test`() {
    val input = tf.placeholder(1 x 16, name = "inputs")
    val W = tf.variable(16 x 4, tf.random_uniform(16 x 4, 0f, 0.01f), name = "W")
    val Qout = tf.matmul(input, W, name = "Qout")
    val predict = tf.argmax(Qout, 1, name = "predict")
    val nextQ = tf.placeholder(1 x 4, name = "nextQ")
    val loss = tf.sum(tf.square(tf.sub(nextQ, Qout)), tf.const(intArrayOf(0, 1)), name = "loss")
    println(tf.debugString())
  }
  
  @Test
  fun `graph def test`() {
    Tensor.create(floatArrayOf(3f, 4f, 5f)).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.DT_FLOAT)
          .setAttr("value", it)
          .build()
    }
    Tensor.create(floatArrayOf(3f, 4f, 5f, 6f, 7f)).use {
      tf.g.opBuilder("Const", "B")
          .setAttr("dtype", DataType.DT_FLOAT)
          .setAttr("value", it)
          .build()
    }
    val s = tf.g.toGraphDef()
    val def = GraphDef.parseFrom(s)
    println(def)
    println(String(s))
  }
  
  
  @Test
  fun `const test`() {
    val _A = Tensor.create(9.3f).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.DT_FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = _A.fetch()
      println(A.floatValue())
    }
  }
  
  @Test
  fun `const array test`() {
    Tensor.create(floatArrayOf(3f, 4f, 5f)).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.DT_FLOAT)
          .setAttr("value", it)
          .build()
    }
    Tensor.create(floatArrayOf(3f, 4f, 5f, 6f, 7f)).use {
      tf.g.opBuilder("Const", "B")
          .setAttr("dtype", DataType.DT_FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      val B = fetch("B")
      val buf = FloatBuffer.allocate(3)
      A.writeTo(buf)
      buf.flip()
      println(buf.remaining())
      while (buf.hasRemaining())
        println(buf.get())
      println(buf.remaining())
    }
  }
  
  @Test
  fun `write 3d tensor`() {
    var i = 0f
    val tensor = Array(2) {
      Array(2) {
        FloatArray(3) {
          i++
        }
      }
    }
    
    Tensor.create(tensor).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.DT_FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      val buf = FloatBuffer.allocate(12)
      A.writeTo(buf)
      buf.flip()
      println(buf.remaining())
      i = 0f
      while (buf.hasRemaining()) {
        val acutal = buf.get()
        println(acutal)
        Assert.assertEquals(i++, acutal)
      }
      println(buf.remaining())
    }
  }
  
  @Test
  fun `copyTo 3d tensor`() {
    var i = 0f
    val tensor = Array(2) {
      Array(2) {
        FloatArray(3) {
          i++
        }
      }
    }
    
    Tensor.create(tensor).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.DT_FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      
      val A_array = Array(2) { Array(2) { FloatArray(3) } }
      A.copyTo(A_array)
      i = 0f
      for (a in A_array)
        for (b in a)
          for (c in b)
            Assert.assertEquals(i++, c)
    }
  }
}