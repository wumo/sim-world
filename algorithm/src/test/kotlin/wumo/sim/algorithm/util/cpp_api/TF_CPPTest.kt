package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.bytedeco.javacpp.tensorflow.DT_INT32
import org.junit.Test

import wumo.sim.algorithm.util.cpp_api.ops.*
import wumo.sim.algorithm.util.dim
import wumo.sim.algorithm.util.x
import wumo.sim.envs.toy_text.FrozenLake
import wumo.sim.util.math.Rand

class TF_CPPTest : BaseTest() {
  @Test
  fun `tensor helper`() {
    val E = tf.const(2 x 3, 9f, "E")
    val F = tf.const(2 x 3, "hello", "F")
    println(tf.debugString())
    tf.session {
      val result = eval<Float>(E)
      println(result[0, 1])
      result[0, 1] = 2f
      println(result[0, 1])
      val result2 = TensorHelper.wrap<Float>(result.nativeTensor)
      println(result2[0, 1])
      
      val result3 = eval<String>(F)
      println(result3[0, 1])
      println(result3[0, 2])
      println(result3[0, 1])
      result3[0, 1] = "hello,world"
      println(result3[0, 2])
      println(result3[0, 1])
      val result4 = TensorHelper.wrap<String>(result3.nativeTensor)
      println(result4[0, 1])
    }
  }
  
  @Test
  fun const() {
    val A = tf.const(1f)
    val B = tf.const(2L, "B")
    val C = tf.const(2 x 2, arrayOf("hello", "world", "fine", "tensorflow"), "C")
    val D = tf.const(2L, scope = tf.root.WithOpName(""))
    val E = tf.const(2 x 3, 9f, "E")
    val F = tf.const(2 x 2, 9.toByte(), "F")
    val G = tf.const(2 x 2, 10L, "G")
    val H = tf.const(2 x 2, true, "H")
    val I = tf.const(2 x 2, floatArrayOf(1f, 2f, 3f, 4f), "I")
    val J = tf.const(2 x 2, arrayOf(1f, 2f, 3f, 4f), "I")
    println(tf.debugString())
    tf.session {
      A.eval()
      val _a = eval<Float>(A)
      println(_a.get())
      B.eval()
      val _b = eval<Long>(B)
      println(_b.get())
      C.eval()
      val _c = eval<String>(C)
      println(_c[0, 0])
      println(_c[0, 1])
      println(_c[1, 0])
      println(_c[1, 1])
      E.eval()
      F.eval()
      val _f = eval<Byte>(F)
      println(_f[1, 1])
      G.eval()
      H.eval()
      val _h = eval<Boolean>(H)
      println(_h[1, 1])
      I.eval()
      J.eval()
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
      val (_a, _b) = eval<Float, Float>(A, B)
      println(_a.get())
      println(_b[0, 1])
      val (_c, _d) = eval(*arrayOf(A, B))
      println(_c.get())
      println(_d[0, 1])
    }
  }
  
  @Test
  fun placeholder() {
    val A = tf.placeholder(2 x 2 x 2)
    val B = tf.variable(2 x 2 x 2, A)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      init.run(A to tensor(2 x 2 x 2, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f))
      B.eval()
      feed(A to tensor(2 x 2 x 2, 1f, 1f, 1f, 4f, 5f, 6f, 7f, 8f))
      target(init)
      run()
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
    val A = tf.const(floatArrayOf(1f, 2f, 3f), name = "A")
    val axis = tf.const(0, name = "axis")
    val sum = tf.sum(A, axis, "sum")
    
    val B = tf.const(1 x 3, floatArrayOf(1f, 2f, 3f), name = "B")
    val axisB = tf.const(intArrayOf(0, 1), name = "axisB")
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
  
  @Test
  fun `gradienet descent`() {
    val x = tf.variable(1f, name = "x")
    val y = tf.variable(1f, name = "y")
    val z = tf.mul(x, y, "z")
    val opt = tf.gradientDescentOptimizer(0.1f, z)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      init.run()
      x.eval()
      y.eval()
      z.eval()
      for (i in 0 until 10) {
        println(i)
        opt.run()
        x.eval()
        y.eval()
        z.eval()
      }
    }
  }
  
  @Test
  fun ` Multi-armed bandit`() {
    //https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
    val weights = tf.variable(dim(4), 1f)
    val chosen_action = tf.argmax(weights, 0)
    
    val reward_holder = tf.placeholder(dim(1), dtype = DT_FLOAT)
    val action_holder = tf.placeholder(dim(1), dtype = DT_INT32)
//    val responsible_weight = t
  }
}