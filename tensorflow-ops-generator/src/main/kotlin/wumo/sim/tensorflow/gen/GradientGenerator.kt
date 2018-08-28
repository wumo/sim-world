package wumo.sim.tensorflow.gen

import org.antlr.v4.runtime.CharStreams
import org.antlr.v4.runtime.CommonTokenStream
import wumo.python3.Python3BaseVisitor
import wumo.python3.Python3Lexer
import wumo.python3.Python3Parser
import wumo.python3.Python3Parser.*
import wumo.sim.util.plusAssign
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
  }.forEach { translate(it, toDir) }
}

val nonDiff = Regex("""ops[.]NotDifferentiable[(]("\w+")[)]""")

fun translate(file: File, toPath: File) {
  File("${toPath.absolutePath}${File.separatorChar}${file.nameWithoutExtension}.kt").sink {
    val sb = StringBuilder()
    sb += """
      import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
      import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
      import wumo.sim.util.append
      fun register_${file.nameWithoutExtension}(){
    """
    val data = readString(file)
    val lexer = Python3Lexer(CharStreams.fromFileName(file.absolutePath))
    val tokenStream = CommonTokenStream(lexer)
    val parser = Python3Parser(tokenStream)
    val root = parser.file_input()
    root.accept(object : Python3BaseVisitor<Python3Parser.File_inputContext>() {
      override fun visitSimple_stmt(ctx: Simple_stmtContext?): File_inputContext {
        nonDiff.find(ctx!!.text)?.let {
          val (opType) = it.destructured
          sb += "registerNonDifferentiable($opType)\n"
        }
        return root
      }
      
      override fun visitDecorated(ctx: DecoratedContext?): File_inputContext {
        val opTypes = ctx!!.decorators().decorator().filter { it.dotted_name().text == "ops.RegisterGradient" }
            .map {
              val args = it.arglist().argument()
              assert(args.size == 1)
              args[0].text
            }
        
        if (opTypes.isEmpty()) return root
        sb += """register(${opTypes.joinToString(", ")}){op,grad,gradOutputs->
              ${handle(ctx.funcdef().suite())}
            }
            """
        return root
      }
      
      override fun visitFuncdef(ctx: FuncdefContext?): File_inputContext {
        ctx!!
        sb += """fun ${ctx.NAME().text}(${ctx.parameters().text}){
            ${handle(ctx.suite())}
          }
        """
        return root
      }
    })
    for (stmt in root.stmt()) {
      stmt.simple_stmt()?.let {
      
      }
      stmt.compound_stmt()?.decorated()?.let {
      
      }
    }
    sb += "}"
    it.writeString(sb.toString())
  }
}

fun handle(funcDef: FuncdefContext): String {
  val sb = StringBuilder()
  sb += """fun ${funcDef.NAME().text}(${funcDef.parameters().text}){
      ${handle(funcDef.suite())}
    }
  """
  return sb.toString()
}

fun handle(suite: SuiteContext): String {
  val sb = StringBuilder()
  val vars = mutableSetOf<String>()
  suite.simple_stmt()?.small_stmt()?.forEach {
    sb += handle(it, vars)
  }
  suite.stmt()?.forEach {
    it?.simple_stmt()?.small_stmt()?.forEach {
      sb += handle(it, vars)
    }
    it?.compound_stmt()?.let {
      sb += handle(it, vars)
    }
  }
  return sb.toString()
}

fun handle(ctx: Small_stmtContext, vars: MutableSet<String>): String {
  val sb = StringBuilder()
  val ctx = ctx.getChild(0)
  when (ctx) {
    is Expr_stmtContext -> sb += handle(ctx, vars)
    is Flow_stmtContext -> {
      val ctx = ctx.getChild(0)
      when (ctx) {
        is Return_stmtContext -> sb += handle(ctx)
        else -> sb += ctx.text + "\n"
      }
    }
    else -> {
      sb += """/* ignored
         ${ctx.text}
         */
         """
    }
  }
  return sb.toString()
}

fun handle(ctx: Compound_stmtContext, vars: MutableSet<String>): String {
  val sb = StringBuilder()
  val ctx = ctx.getChild(0)
  
  when (ctx) {
    is If_stmtContext -> {
      val tests = ctx.test()
      val suites = ctx.suite()
      
      if (tests.size == 1) {
        sb += """if(${handle(tests[0])}){
            ${handle(suites[0])}
          }
        """
        if (suites.size > tests.size) {
          //else branch
          sb += """else{
              ${handle(suites.last())}
            }
          """
        }
      } else {
        sb += """when{
        """
        tests.zip(suites).forEach { (test, suite) ->
          sb += """${handle(test)} -> {
              ${handle(suite)}
            }
          """
        }
        if (suites.size > tests.size) {
          //else branch
          sb += """else -> {
              ${handle(suites.last())}
            }
          """
        }
        sb += "}\n"
      }
    }
    is For_stmtContext -> {
      
      ctx
    }
    is While_stmtContext -> {
      ctx
    }
    is With_itemContext -> {
      ctx
    }
    is FuncdefContext -> {
      sb += handle(ctx)
    }
    else -> {
      sb += ctx.text + "\n"
    }
  }
  return sb.toString()
}

fun handle(ctx: Return_stmtContext): String {
  val sb = StringBuilder()
  sb += handle(ctx.testlist())
//  sb += "gradOutputs.append(${ctx.testlist().text})\n"
  return sb.toString()
}

fun handle(ctx: Expr_stmtContext, vars: MutableSet<String>): String {
  val sb = StringBuilder()
  if (ctx.childCount == 3 && ctx.getChild(1).text == "=") {//assign
    val varList = (ctx.getChild(0) as Testlist_star_exprContext).test()
    if (varList.size > 1)
      sb += "var (${varList.joinToString(",")}("
    else {
      if (varList[0].text !in vars) {
        sb += "var "
        vars += varList[0].text
      }
      sb += varList[0].text
    }
    sb += "="
    sb += ctx.getChild(2).text + "\n"
  } else
    sb += ctx.text + "\n"
  return sb.toString()
}