package wumo.sim.tensorflow.samples

import org.junit.Test
import wumo.sim.envs.toy_text.FrozenLake
import wumo.sim.tensorflow.ops.BaseTest
import wumo.sim.tensorflow.ops.basic.minus
import wumo.sim.tensorflow.ops.training.GradientDescentOptimizer
import wumo.sim.tensorflow.tf
import wumo.sim.util.Rand
import wumo.sim.util.Shape
import wumo.sim.util.f
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.types.NDFloat

class `Q-Learning with Neural Networks test` : BaseTest() {
  @Test
  fun test() {
    //https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
    val inputs = tf.placeholder(Shape(1, 16), name = "inputs1")
    val W = tf.variable(tf.randomUniform(Shape(16, 4), 0f, 0.01f), name = "W")
    val Qout = tf.matMul(inputs, W.toOutput(), name = "Qout")
    val predict = tf.argmax(Qout, 1, name = "predict")
    val nextQ = tf.placeholder(Shape(1, 4), name = "nextQ")
    
    val loss = tf.sum(tf.square(nextQ - Qout), name = "loss")
    val optimizer = GradientDescentOptimizer(learningRate = { 0.1f }, name = "train")
    val train = optimizer.minimize(loss)
    val init = tf.globalVariablesInitializer()
    printGraph()
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
          feed(inputs to NDArray(Shape(1, 16), NDFloat) { if (it == s) 1f else 0f })
          val (a, allQ) = eval<Int, Float>(predict, Qout)
          if (Rand().nextDouble() < e)
            a[0] = env.action_space.sample()
          val (s1, r, d) = env.step(a[0])
          feed(inputs to NDArray(Shape(1, 16), NDFloat) { if (it == s1) 1f else 0f })
          val Q1 = eval<Float>(Qout)
          val maxQ1 = Q1.max()!!
          val targetQ = allQ
          
          targetQ[0, a[0]] = (r + y * maxQ1).toFloat()
          
          feed(inputs to NDArray(Shape(1, 16), NDFloat) { if (it == s) 1f else 0f },
               nextQ to targetQ)
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