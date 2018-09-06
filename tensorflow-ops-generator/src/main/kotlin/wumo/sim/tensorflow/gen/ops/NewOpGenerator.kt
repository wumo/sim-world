package wumo.sim.tensorflow.gen.ops

import org.tensorflow.framework.OpDef
import wumo.sim.util.t2

class NewOpGenerator(opDef: OpDef, val genObject: String) : OpGenerator(opDef) {
  val sb = StringBuilder()
  
  fun generateOpFunctions(): t2<String, String> {
    val parts = generateOp()
    val (kotlinArguments, buildFunc, returnType, addInput, addAttr, inputs) = parts
    val genOp = defFunction(parts)
    
    val op = """fun $name($kotlinArguments):$returnType{
      return $genObject.$name(${inputs.joinToString(", ")})
    }
    """
    return t2(genOp, op)
  }
}