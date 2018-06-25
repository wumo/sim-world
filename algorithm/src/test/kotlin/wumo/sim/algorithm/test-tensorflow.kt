package wumo.sim.algorithm


import java.io.IOException
import java.io.PrintStream
import java.nio.charset.Charset
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Arrays
import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.types.UInt8

/** Sample use of the TensorFlow Java API to label images using a pre-trained model.  */
private fun printUsage(s: PrintStream) {
  val url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
  s.println(
      "Java program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)")
  s.println("to label JPEG images.")
  s.println("TensorFlow version: " + TensorFlow.version())
  s.println()
  s.println("Usage: label_image <model dir> <image file>")
  s.println()
  s.println("Where:")
  s.println("<model dir> is a directory containing the unzipped contents of the inception model")
  s.println("            (from $url)")
  s.println("<image file> is the path to a JPEG image file")
}

fun main(args: Array<String>) {
  if (args.size != 2) {
    printUsage(System.err)
    System.exit(1)
  }
  val modelDir = args[0]
  val imageFile = args[1]
  
  val graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"))
  val labels = readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"))
  val imageBytes = readAllBytesOrExit(Paths.get(imageFile))
  
  constructAndExecuteGraphToNormalizeImage(imageBytes).use { image ->
    val labelProbabilities = executeInceptionGraph(graphDef, image)
    val bestLabelIdx = maxIndex(labelProbabilities)
    println(
        String.format("BEST MATCH: %s (%.2f%% likely)",
            labels!![bestLabelIdx],
            labelProbabilities[bestLabelIdx] * 100f))
  }
}

private fun constructAndExecuteGraphToNormalizeImage(imageBytes: ByteArray?): Tensor<Float> {
  Graph().use { g ->
    val b = GraphBuilder(g)
    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    val H = 224
    val W = 224
    val mean = 117f
    val scale = 1f
    
    // Since the graph is being constructed once per execution here, we can use a constant for the
    // input image. If the graph were to be re-used for multiple input images, a placeholder would
    // have been more appropriate.
    val input = b.constant("input", imageBytes)
    val output = b.div(
        b.sub(
            b.resizeBilinear(
                b.expandDims(
                    b.cast(b.decodeJpeg(input, 3), Float::class.java),
                    b.constant("make_batch", 0)),
                b.constant("size", intArrayOf(H, W))),
            b.constant("mean", mean)),
        b.constant("scale", scale))
    Session(g).use { s ->
      // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
      return s.runner().fetch(output.op().name()).run()[0].expect(Float::class.java)
    }
  }
}

private fun executeInceptionGraph(graphDef: ByteArray?, image: Tensor<Float>): FloatArray {
  Graph().use { g ->
    g.importGraphDef(graphDef!!)
    Session(g).use { s ->
      s.runner().feed("input", image).fetch("output").run()[0].expect(Float::class.java).use { result ->
        val rshape = result.shape()
        if (result.numDimensions() != 2 || rshape[0] != 1L) {
          throw RuntimeException(
              String.format(
                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                  Arrays.toString(rshape)))
        }
        val nlabels = rshape[1].toInt()
        return result.copyTo(Array(1) { FloatArray(nlabels) })[0]
      }
    }// Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
  }
}

private fun maxIndex(probabilities: FloatArray): Int {
  var best = 0
  for (i in 1 until probabilities.size) {
    if (probabilities[i] > probabilities[best]) {
      best = i
    }
  }
  return best
}

private fun readAllBytesOrExit(path: Path): ByteArray? {
  try {
    return Files.readAllBytes(path)
  } catch (e: IOException) {
    System.err.println("Failed to read [" + path + "]: " + e.message)
    System.exit(1)
  }
  
  return null
}

private fun readAllLinesOrExit(path: Path): List<String>? {
  try {
    return Files.readAllLines(path, Charset.forName("UTF-8"))
  } catch (e: IOException) {
    System.err.println("Failed to read [" + path + "]: " + e.message)
    System.exit(0)
  }
  
  return null
}

// In the fullness of time, equivalents of the methods of this class should be auto-generated from
// the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
// like Python, C++ and Go.
internal class GraphBuilder(private val g: Graph) {
  
  fun div(x: Output<Float>, y: Output<Float>): Output<Float> {
    return binaryOp("Div", x, y)
  }
  
  fun <T> sub(x: Output<T>, y: Output<T>): Output<T> {
    return binaryOp("Sub", x, y)
  }
  
  fun <T> resizeBilinear(images: Output<T>, size: Output<Int>): Output<Float> {
    return binaryOp3("ResizeBilinear", images, size)
  }
  
  fun <T> expandDims(input: Output<T>, dim: Output<Int>): Output<T> {
    return binaryOp3("ExpandDims", input, dim)
  }
  
  fun <T, U> cast(value: Output<T>, type: Class<U>): Output<U> {
    val dtype = DataType.fromClass(type)
    return g.opBuilder("Cast", "Cast")
        .addInput(value)
        .setAttr("DstT", dtype)
        .build()
        .output(0)
  }
  
  fun decodeJpeg(contents: Output<String>, channels: Long): Output<UInt8> {
    return g.opBuilder("DecodeJpeg", "DecodeJpeg")
        .addInput(contents)
        .setAttr("channels", channels)
        .build()
        .output(0)
  }
  
  fun <T> constant(name: String, value: Any?, type: Class<T>): Output<T> {
    Tensor.create(value!!, type).use { t ->
      return g.opBuilder("Const", name)
          .setAttr("dtype", DataType.fromClass(type))
          .setAttr("value", t)
          .build()
          .output(0)
    }
  }
  
  fun constant(name: String, value: ByteArray?): Output<String> {
    return this.constant(name, value, String::class.java)
  }
  
  fun constant(name: String, value: Int): Output<Int> {
    return this.constant(name, value, Int::class.java)
  }
  
  fun constant(name: String, value: IntArray): Output<Int> {
    return this.constant(name, value, Int::class.java)
  }
  
  fun constant(name: String, value: Float): Output<Float> {
    return this.constant(name, value, Float::class.java)
  }
  
  private fun <T> binaryOp(type: String, in1: Output<T>, in2: Output<T>): Output<T> {
    return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0)
  }
  
  private fun <T, U, V> binaryOp3(type: String, in1: Output<U>, in2: Output<V>): Output<T> {
    return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0)
  }
}