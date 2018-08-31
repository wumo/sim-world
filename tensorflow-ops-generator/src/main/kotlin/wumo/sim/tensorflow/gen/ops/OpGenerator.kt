package wumo.sim.tensorflow.gen.ops

import com.google.protobuf.TextFormat
import org.tensorflow.framework.AttrValue
import org.tensorflow.framework.OpDef
import renames
import wumo.sim.tensorflow.gen.toCamelCase
import java.util.*

class OpGenerator(val opDef: OpDef, val sb: StringBuilder) {
  val name = processName(opDef.name).let { n -> if (n.startsWith("_")) n else n }
      .let { name ->
        if (renames.any { name.equals(it, true) })
          "_$name"
        else name
      }
  val argumentTypes = hashMapOf<String, String>()
  val inputs = mutableListOf<Pair<String, String>>()
  val inputsRef = hashMapOf<String, Boolean>()
  val parameters = mutableListOf<Pair<String, String>>()
  val parameterDefaults = hashMapOf<String, AttrValue>()
  val inferrableAttributes = hashMapOf<String, String>()
  var numOutputs: Int = 0
  
  init {
    initialize()
  }
  
  fun generateOpFunction() {
    val arguments = inputs + parameters
    
    var kotlinArguments = arguments.joinToString(", ") { (name, kotlinName) ->
      val paramType = typeToKotlinType[argumentTypes[name]]!!
      val defaultValue = if (name in parameterDefaults) {
        val value = attrValueToKotlin(argumentTypes[name]!!, parameterDefaults[name]!!)
        if (value != null) " = $value" else ""
      } else ""
      "${kotlinName.toCamelCase(false)}: $paramType$defaultValue"
    }.trim()
    kotlinArguments += "${if (kotlinArguments.isNotBlank()) ", " else ""}name: String = \"${opDef.name}\""
    val addInput = inputs.joinToString("\n") { (name, kotlinName) ->
      "addInput(${kotlinName.toCamelCase(false)},${inputsRef[name]})"
    }
    val addAttr = parameters.joinToString("\n") { (name, kotlinName) ->
      "attr(\"$name\", ${kotlinName.toCamelCase(false)})"
    }
    val buildFunc = when (numOutputs) {
      0 -> "buildOp"
      1 -> "buildOpTensor"
      else -> "buildOpTensors"
    }
    with(sb) {
      append("fun $name($kotlinArguments) = run {\n")
      append("$buildFunc(\"${opDef.name}\", name){\n")
      if (addInput.isNotBlank()) append(addInput).append('\n')
      if (addAttr.isNotBlank()) append(addAttr).append('\n')
      append("}\n}\n")
    }
  }
  
  fun initialize() {
    // Process input arguments.
    opDef.inputArgList.withIndex().forEach { (index, arg) ->
      argumentTypes[arg.name] = if (arg.numberAttr.isNotEmpty()) "list(Output)" else "Output"
      inputsRef[arg.name] = arg.isRef
      val inferrableAttrs = mutableListOf<Pair<String, String>>()
      inferrableAttrs += if (arg.typeAttr.isNotEmpty())
        arg.typeAttr to "type"
      else
        arg.typeListAttr to "list(type)"
      if (!arg.numberAttr.isEmpty())
        inferrableAttrs += arg.numberAttr to "int"
      inferrableAttrs.forEach { a ->
        argumentTypes[a.first] = a.second
        inferrableAttributes.getOrPut(a.first) { opDef.getInputArg(index).name }
      }
      inputs += arg.name to processName(arg.name)
    }
    
    // Process attributes that have not been inferred. We do not want add inferred attributes to the Scala function
    // signatures.
    val attrsWithoutDefaults = mutableListOf<String>()
    val attrsWithDefaults = mutableListOf<String>()
    opDef.attrList.filter { it.name !in inferrableAttributes }
        .forEach { attr ->
          argumentTypes[attr.name] = attr.type
          if (attr.hasDefaultValue()) {
            parameterDefaults[attr.name] = attr.defaultValue
            attrsWithDefaults += attr.name
          } else
            attrsWithoutDefaults += attr.name
        }
    // Save the list of attribute parameters (i.e., attributes that won't be inferred). Those with defaults go at the
    // end. Get the attributes in the order we want by taking the attributes without defaults from the end of
    // argsWithoutDefaults, and then adding argsWithDefaults.
    attrsWithoutDefaults.forEach { parameters += it to processName(it) }
    attrsWithDefaults.forEach { parameters += it to processName(it) }
    
    // Create an expression that computes the number of outputs of this op.
    // If output i is list output, outputSizes[i] will be set to a string with the Scala expression that will evaluate
    // to its length. outputSizes[i] is empty for non-list outputs.
    var numOutputs = 0
    opDef.outputArgList.forEach { outputArg ->
      when {
        outputArg.numberAttr.isNotEmpty() -> {
          numOutputs = 2//fake number, indicate multiple outputs
        }
        
        outputArg.typeListAttr.isNotEmpty() -> {
          numOutputs = 2//fake number, indicate multiple outputs
        }
        
        else -> {
          numOutputs++
        }
      }
    }
    this.numOutputs = numOutputs
  }
  
  companion object {
    val reservedKeywords = setOf(
        "package", "as", "type", "class", "this", "val", "var", "fun", "extension", "for",
        "null", "typeof", "new", "true", "false", "is", "in", "throw", "return", "break", "continue",
        "object", "if", "else", "while", "do", "when", "out", "ref", "try",
        "by", "get", "set", "import", "final", "abstract", "enum",
        "open", "annotation", "override", "private", "public", "internal",
        "protected", "catch", "finally")
    
    fun processName(name: String) = run {
      var processedName = name.decapitalize()
      if (processedName in reservedKeywords) processedName = "_$processedName"
      processedName
    }
    
    val typeToKotlinType = mapOf(
        "func" to "NameAttrList",
        "list(func)" to "List<NameAttrList>",
        "string" to "String",
        "int" to "Long",
        "float" to "Float",
        "bool" to "Boolean",
        "type" to "DataType<*>",
        "shape" to "Shape",
        "Output" to "Output",
        "tensor" to "NDArray<*>",
        "list(string)" to "Array<String>",
        "list(int)" to "Array<Long>",
        "list(float)" to "Array<Float>",
        "list(bool)" to "Array<Boolean>",
        "list(type)" to "Array<Long>",
        "list(shape)" to "Array<Shape>",
        "list(tensor)" to "Array<NDArray<*>>",
        "list(Output)" to "List<Output>"
    )
    
    /**Converts the provided TensorFlow attribute
     *  value to a string representing the same value in Kotlin*/
    fun attrValueToKotlin(attr_type: String, value: AttrValue) = run {
      when (attr_type) {
        "string" -> "${'"'}${escapeString(value.s.toStringUtf8())}${'"'}"
        "int" -> "${value.i}L"
        
        "float" -> {
          val f = value.f
          when (f) {
            Float.NaN -> "Float.NaN"
            Float.NEGATIVE_INFINITY -> "Float.NEGATIVE_INFINITY"
            Float.POSITIVE_INFINITY -> "Float.POSITIVE_INFINITY"
            else -> "${f}f"
          }
        }
        
        "bool" -> value.b.toString()
        "type" -> value.type.name.substring(3)
        
        "shape" -> {
          val s = value.shape
          if (s.unknownRank) "Shape()"
          else "Shape(longArrayOf(${value.shape.dimList.joinToString(", ")}))"
        }
        
        "tensor" -> {
//          "${"\"\"\""}${TextFormat.shortDebugString(value.tensor)}${"\"\"\""}"
          null//TODO
        }
        
        "func" -> null//"${'"'}${escapeString(value.func.name)}${'"'}"
        else -> {
          if (attr_type.startsWith("list(")) {
            val content = when {
              value.list.sCount > 0 ->
                value.list.sList.joinToString(", ") {
                  "${'"'}${escapeString(it.toStringUtf8())}${'"'}"
                }
              value.list.iCount > 0 ->
                value.list.iList.joinToString(", ") { "${it}L" }
              value.list.fCount > 0 ->
                value.list.fList.joinToString(", ") {
                  when (it) {
                    Float.NaN -> "Float.NaN"
                    Float.NEGATIVE_INFINITY -> "Float.NEGATIVE_INFINITY"
                    Float.POSITIVE_INFINITY -> "Float.POSITIVE_INFINITY"
                    else -> "${it}f"
                  }
                }
              value.list.bCount > 0 ->
                value.list.bList.joinToString(", ") { "$it" }
              value.list.typeCount > 0 ->
                value.list.typeList.joinToString(", ") { "${it.number}" }
              value.list.shapeCount > 0 ->
                value.list.shapeList.joinToString(", ") {
                  if (it.unknownRank) "Shape()"
                  else "Shape(longArrayOf(${it.dimList.joinToString(", ")}))"
                }
              value.list.tensorCount > 0 ->
                "arrayOf(" + value.list.tensorList.joinToString(", ") {
                  "${"\"\"\""}${TextFormat.shortDebugString(it)}${"\"\"\""}"
                } + ")"
              value.list.funcCount > 0 ->
                value.list.funcList.joinToString(", ") { "${'"'}${escapeString(it.name)}${'"'}" }
              else -> ""
            }
            "arrayOf($content)"
          } else ""
        }
      }
    }
    
    fun escapeString(string: String): String = run {
      fun hex(char: Char): String = Integer.toHexString(char.toInt()).toUpperCase(Locale.ENGLISH)
      
      val writer = StringBuilder(string.length * 2)
      val size: Int = string.length
      var i = 0
      while (i < size) {
        val c = string[i]
        when {
          c == '\b' -> {
            writer.append('\\'); writer.append('b')
          }
          
          c == '\n' -> {
            writer.append('\\'); writer.append('n')
          }
          
          c == '\t' -> {
            writer.append('\\'); writer.append('t')
          }
//            '\f' -> {writer.append('\\'); writer.append('f')}
          c == '\r' -> {
            writer.append('\\'); writer.append('r')
          }
          
          c == '\'' -> {
            writer.append('\\'); writer.append('\'')
          }
          
          c == '\"' -> {
            writer.append('\\'); writer.append('\"')
          }
          
          c == '\\' -> {
            writer.append('\\'); writer.append('\\')
          }
          
          c > 0xfff.toChar() -> writer.append("\\u" + hex(c))
          c > 0xff.toChar() -> writer.append("\\u0" + hex(c))
          c > 0x7f.toChar() -> writer.append("\\u00" + hex(c))
          c > 0xf.toChar() && c < 32.toChar() -> writer.append("\\u00" + hex(c))
          c < 32.toChar() -> writer.append("\\u000" + hex(c))
          else -> writer.append(c)
        }
        i += 1
      }
      writer.toString()
    }
  }
}