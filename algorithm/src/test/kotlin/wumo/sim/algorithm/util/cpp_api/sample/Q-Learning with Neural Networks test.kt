package wumo.sim.algorithm.util.cpp_api.sample

import org.junit.Test
import wumo.sim.algorithm.util.cpp_api.BaseTest
import wumo.sim.algorithm.util.cpp_api.ops.*
import wumo.sim.algorithm.util.cpp_api.tensor
import wumo.sim.algorithm.util.x
import wumo.sim.envs.toy_text.FrozenLake
import wumo.sim.util.math.Rand

class `Q-Learning with Neural Networks test` : BaseTest() {
  @Test
  fun test() {
    //https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
    val inputs = tf.placeholder(1 x 16, name = "inputs1")
    val W = tf.variable(16 x 4, tf.random_uniform(16 x 4, 0f, 0.01f), name = "W")
    val Qout = tf.matmul(inputs, W, name = "Qout")
    val predict = tf.argmax(Qout, tf.const(1), name = "predict")
    val nextQ = tf.placeholder(1 x 4, name = "nextQ")
    
    val loss = tf.sum(tf.square(tf.sub(nextQ, Qout)), tf.const(intArrayOf(0, 1)))
    val train = tf.gradientDescentOptimizer(0.1f, loss, "train")
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      init.run()
      val env = FrozenLake()
      val y = .99
      var e = 0.1
      val num_episodes = 2000
      var sum = 0.0
      for (i in 0 until num_episodes) {
        var s = env.reset()
        var rAll = 0.0
        var j = 0
        while (j < 99) {
          j++
          feed(inputs to tensor(1 x 16, *FloatArray(16) { if (it == s) 1f else 0f }))
          val (a, allQ) = eval<Int, Float>(predict, Qout)
          if (Rand().nextDouble() < e)
            a[0] = env.action_space.sample()
          val (s1, r, d) = env.step(a[0])
          feed(inputs to tensor(1 x 16, *FloatArray(16) { if (it == s1) 1f else 0f }))
          val Q1 = eval<Float>(Qout)
          val maxQ1 = Q1.max()!!
          val targetQ = allQ
          targetQ[0, a[0]] = (r + y * maxQ1).toFloat()
          
          feed(inputs to tensor(1 x 16, *FloatArray(16) { if (it == s) 1f else 0f }),
              nextQ to tensor(targetQ))
          train.run()
          rAll += r
          s = s1
          if (d)
            e = 1.0 / (i / 50 + 10)
        }
        println("$rAll-$i")
        sum += rAll
      }
      println(sum / num_episodes)
    }
  }
}