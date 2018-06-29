package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.junit.Test

import org.junit.Before
import wumo.sim.algorithm.util.cpp_api.ops.*
import wumo.sim.algorithm.util.dim
import wumo.sim.algorithm.util.x

class TF_CPPTest {
  lateinit var tf: TF_CPP
  @Before
  fun setup() {
    tf = TF_CPP()
  }
  
  @Test
  fun const() {
    val A = tf.const(1f)
    val B = tf.const(1f, "B")
    val C = tf.const("hello,world", "C")
    val D = tf.const(2L, scope = tf.root.WithOpName(""))
    val E = tf.const(2 x 3, 9f, "E")
    val F = tf.const(2 x 2, 9.0, "F")
    val G = tf.const(2 x 2, 10L, "G")
    val H = tf.const(2 x 2, false, "H")
    val I = tf.const(2 x 2, floatArrayOf(1f, 2f, 3f, 4f), "I")
    println(tf.debugString())
    tf.session {
      A.eval()
      B.eval()
      C.eval()
      E.eval()
      F.eval()
      G.eval()
      H.eval()
      I.eval()
    }
  }
  
  @Test
  fun variable() {
    val A = tf.variable(1f, "A")
    val B = tf.variable(2 x 2, 2f, "B")
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      init.run()
      A.eval()
      B.eval()
    }
  }
  
  @Test
  fun placeholder() {
    val A = tf.placeholder(2 x 2 x 2)
    val B = tf.variable(2 x 2 x 2, A)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      init.run(A to tensor(2 x 2 x 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)))
      B.eval()
    }
  }
  
  @Test
  fun matmul() {
    val A = tf.const(2 x 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), "A")
    val B = tf.const(3 x 2, floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f), "B")
    val matmul = tf.matmul(A, B, "matmul")
    println(tf.debugString())
    tf.session {
      matmul.eval()
    }
  }
  
  @Test
  fun argmax() {
    val A = tf.const(2 x 3, floatArrayOf(3f, 2f, 3f, 1f, 5f, 6f), "A")
    val dim = tf.const(0, "dim")
    val argmax = tf.argmax(A, dim, "argmax")
    println(tf.debugString())
    tf.session {
      argmax.eval()
    }
  }
  
  @Test
  fun sum() {
    val A = tf.const(dim(3), floatArrayOf(1f, 2f, 3f), name = "A")
    val axis = tf.const(dim(1), intArrayOf(0), name = "axis")
    val sum = tf.sum(A, axis, "sum")
    
    val B = tf.const(1 x 3, floatArrayOf(1f, 2f, 3f), name = "B")
    val axisB = tf.const(dim(2), intArrayOf(0, 1), name = "axisB")
    val sumB = tf.sum(B, axisB, name = "sumB")
    println(tf.debugString())
    tf.session {
      sum.eval()
      sumB.eval()
    }
  }
  
  @Test
  fun square() {
    val A = tf.const(2 x 1, floatArrayOf(1f, 2f), "A")
    val square = tf.square(A)
    println(tf.debugString())
    tf.session {
      square.eval()
    }
  }
  
  @Test
  fun `add subtract mul div`() {
    val A = tf.const(2 x 1, 100f, "A")
    val B = tf.const(2f, "B")
    val `A+B` = tf.add(A, B)
    val `A-B` = tf.sub(A, B)
    val `A*B` = tf.mul(A, B)
    val `AdivB` = tf.div(A, B)
    
    println(tf.debugString())
    tf.session {
      `A+B`.eval()
      `A-B`.eval()
      `A*B`.eval()
      `AdivB`.eval()
    }
  }
  
  @Test
  fun random_uniform() {
    val A = tf.random_uniform(2 x 1, DT_FLOAT, "A")
    val B = tf.random_uniform(2 x 1, 1f, 4f, "B")
    println(tf.debugString())
    tf.session {
      A.eval()
      B.eval()
    }
  }
  
  init {
  }
}