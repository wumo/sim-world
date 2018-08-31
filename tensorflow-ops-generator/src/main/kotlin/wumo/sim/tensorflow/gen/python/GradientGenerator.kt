package wumo.sim.tensorflow.gen.python

import org.antlr.v4.runtime.CharStreams
import org.antlr.v4.runtime.CommonTokenStream
import wumo.python3.Python3Lexer
import wumo.python3.Python3Parser
import wumo.sim.util.readString
import wumo.sim.util.sink
import wumo.sim.util.writeString
import java.io.File

fun main(args: Array<String>) {
  generate("tensorflow-ops-generator/resources/tensorflow/tensorflow/python/ops",
           "tensorflow/src/main/kotlin/wumo/sim/tensorflow/ops/gradients/gen")
}

fun generate(fromPath: String,
             toPath: String) {
  val toDir = File(toPath)
  val files = File(fromPath).listFiles { file ->
    file.nameWithoutExtension.endsWith("_grad")
        && file.nameWithoutExtension !in exceptionalFiles
  }.forEach { translate(it, toDir) }
}

val exceptionalFiles = setOf("math_grad", "state_grad")

fun translate(file: File, toPath: File) {
  File("${toPath.absolutePath}${File.separatorChar}${file.nameWithoutExtension}.kt").sink { fileOut ->
    val data = readString(file)
    val lexer = Python3Lexer(CharStreams.fromFileName(file.absolutePath))
    val tokenStream = CommonTokenStream(lexer)
    val parser = Python3Parser(tokenStream)
    val file_input = parser.file_input()
    val visitor = Visitor(file.nameWithoutExtension)
    val str = visitor.visit(file_input)
    fileOut.writeString(str!!)
  }
}