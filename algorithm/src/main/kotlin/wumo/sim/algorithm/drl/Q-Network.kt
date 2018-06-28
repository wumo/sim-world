package wumo.sim.algorithm.drl

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.bytedeco.javacpp.tensorflow.Scope.NewRootScope
import wumo.sim.algorithm.util.TF_CPP
import wumo.sim.algorithm.util.x
import wumo.sim.envs.toy_text.FrozenLake
import wumo.sim.util.math.Rand
import java.nio.FloatBuffer
import java.nio.LongBuffer

fun main(args: Array<String>) {
  // Load all javacpp-preset classes and nativeGraph libraries
  Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
  
  // Platform-specific initialization routine
  tensorflow.InitMain("trainer", null as IntArray?, null)
  
  val scope = NewRootScope()
  val tf = TF_CPP(scope)
  val inputs = tf.placeholder(1 x 16, name = "inputs1")
  val W = tf.variable(16 x 4, tf.random_uniform(16 x 4, 0f, 0.01f), name = "W")
  val Qout = tf.matmul(inputs, W, name = "Qout")
  val predict = tf.argmax(Qout, tf.const(1), name = "predict")
  val nextQ = tf.placeholder(1 x 4, name = "nextQ")
  
  val loss = tf.sum(tf.square(tf.subtract(nextQ, Qout)), tf.tensor(0, 1))
  tf.GradientDescentOptimizer(0.1f, loss, "apply")
  
  tf.session {
    val outputs = TensorVector()
    Run(StringTensorPairVector(), StringVector(), StringVector("W/assign"), outputs)
    
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
        Run(StringTensorPairVector(arrayOf("inputs1"),
            arrayOf(Tensor.create(FloatArray(16) { if (it == s) 1f else 0f }, TensorShape(1, 16)))),
            StringVector("predict", "Qout"),
            StringVector(),
            outputs)
        val size = outputs.size()
        val a = outputs[0]
        val allQ = FloatArray(4)
        outputs[1].createBuffer<FloatBuffer>().get(allQ)
        var _a = a.createBuffer<LongBuffer>()[0].toInt()
        if (Rand().nextDouble() < e)
          _a = env.action_space.sample()
        val (s1, r, d) = env.step(_a)
        outputs.clear()
        Run(StringTensorPairVector(arrayOf("inputs1"),
            arrayOf(Tensor.create(FloatArray(16) { if (it == s1) 1f else 0f }, TensorShape(1, 16)))),
            StringVector("Qout"),
            StringVector(),
            outputs)
        val Q1 = FloatArray(4)
        outputs[0].createBuffer<FloatBuffer>().get(Q1)
        
        val maxQ1 = Q1.max()!!
        val targetQ = allQ
        targetQ[_a] = (r + y * maxQ1).toFloat()
        
        outputs.clear()
        Run(StringTensorPairVector(arrayOf("inputs1", "nextQ"),
            arrayOf(Tensor.create(FloatArray(16) { if (it == s) 1f else 0f }, TensorShape(1, 16)),
                Tensor.create(targetQ, TensorShape(1, 4)))),
            StringVector(),
            StringVector("W/apply"),
            outputs)
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
