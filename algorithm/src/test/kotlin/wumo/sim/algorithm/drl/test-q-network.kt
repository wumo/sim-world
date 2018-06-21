package wumo.sim.algorithm.drl

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.envs.toy_text.FrozenLake
import wumo.sim.util.math.Rand
import java.nio.FloatBuffer
import java.nio.LongBuffer

fun main(args: Array<String>) {
  Loader.load(tensorflow::class.java)
  
  // Platform-specific initialization routine
  tensorflow.InitMain("trainer", null as IntArray?, null)
  val def = GraphDef()
//  val loc = ResourceLoader.getResource("def.pb")
  TF_CHECK_OK(tensorflow.ReadTextProto(Env.Default(), "resources/q-network.pbtxt", def))
  val session = Session(SessionOptions())
  SetDefaultDevice("/gpu:0", def)
  TF_CHECK_OK(session.Create(def))
  val outputs = TensorVector()
  session.Run(StringTensorPairVector(), StringVector(), StringVector("init"), outputs)
  
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
      session.Run(StringTensorPairVector(arrayOf("inputs1"),
          arrayOf(Tensor.create(FloatArray(16) { if (it == s) 1f else 0f }, TensorShape(1, 16)))),
          StringVector("predict", "Qout"),
          StringVector(),
          outputs)
      val a = outputs[0]
      val allQ = FloatArray(4)
      outputs[1].createBuffer<FloatBuffer>().get(allQ)
      var _a = a.createBuffer<LongBuffer>()[0].toInt()
      if (Rand().nextDouble() < e)
        _a = env.action_space.sample()
      val (s1, r, d) = env.step(_a)
      outputs.clear()
      session.Run(StringTensorPairVector(arrayOf("inputs1"),
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
      session.Run(StringTensorPairVector(arrayOf("inputs1", "nextQ"),
          arrayOf(Tensor.create(FloatArray(16) { if (it == s) 1f else 0f }, TensorShape(1, 16)),
              Tensor.create(targetQ, TensorShape(1, 4)))),
          StringVector(),
          StringVector("train"),
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
  session.close()
  def.close()
}