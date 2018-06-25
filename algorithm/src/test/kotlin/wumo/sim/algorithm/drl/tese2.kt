package wumo.sim.algorithm.drl

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow.*
import java.nio.IntBuffer
import org.tensorflow.Session.Run
import org.bytedeco.javacpp.tensorflow.Scope.NewRootScope
import java.nio.FloatBuffer


fun main(args: Array<String>) {
  
  // Load all javacpp-preset classes and nativeGraph libraries
  Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
  
  // Platform-specific initialization routine
  InitMain("trainer", null as IntArray?, null)
  
  // Create a new empty graph
  val scope = Scope.NewRootScope()
  
  // (3,2) matrix of ones and sixes
  val shape = TensorShape(3, 2)
  val ones = Const(scope.WithOpName("ones"), 1, shape)
  val sixes = Const(scope.WithOpName("sixes"), 6, shape)
  
  // Vertical concatenation of those matrices
  val ov = OutputVector(ones, sixes)
  val inputList = InputList(ov)
  val axis = Input(Const(scope.WithOpName("axis"), 0))
  val concat = Concat(scope.WithOpName("concat"), inputList, axis)
  
  // Build a graph definition object
  val def = GraphDef()
  TF_CHECK_OK(scope.ToGraphDef(def))
  
  // Creates a session.
  val options = SessionOptions()
  Session(options).use { session ->
    
    // Create the graph to be used for the session.
    TF_CHECK_OK(session.Create(def))
    
    // Input and output of a single session run.
    val input_feed = StringTensorPairVector()
    val output_tensor_name = StringVector("concat:0")
    val target_tensor_name = StringVector()
    val outputs = TensorVector()
    
    // Run the session once
    TF_CHECK_OK(session.Run(input_feed, output_tensor_name, target_tensor_name, outputs))
    
    // Print the concatenation output
    for (output in outputs.get()) {
      val y_flat = output.createBuffer<IntBuffer>()
      for (i in 0 until output.NumElements())
        println(y_flat[i.toInt()])
    }
  }
}