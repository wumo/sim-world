package wumo.sim.tensorflow.gen

import java.io.File

fun main(args: Array<String>) {
  generate("tensorflow-ops-generator/resources/tensorflow/tensorflow/python/ops")
}

fun generate(path: String) {
  val files = File(path).listFiles { file ->
    file.nameWithoutExtension.endsWith("_grad")
  }
  files.forEach { println(it) }
}