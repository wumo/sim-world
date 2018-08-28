package wumo.sim.tensorflow.gen

import org.antlr.v4.runtime.CharStreams
import org.antlr.v4.runtime.CommonTokenStream
import wumo.python3.Python3BaseVisitor
import wumo.python3.Python3Lexer
import wumo.python3.Python3Parser
import wumo.python3.Python3Parser.*
import wumo.sim.util.*
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
  if (file.nameWithoutExtension != "math_grad") return
  File("${toPath.absolutePath}${File.separatorChar}${file.nameWithoutExtension}.kt").sink {
    val sb = StringBuilder()
    sb += """
      import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
      import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
      import wumo.sim.util.append
      import wumo.sim.tensorflow.tf
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
        sb += """register(${opTypes.joinToString(", ")}){op,grad->
              ${handle(ctx.funcdef(), CTX(isLambda = true))}
            }
            """
        return root
      }
      
      override fun visitFuncdef(funcDef: FuncdefContext): File_inputContext {
        funcDef!!
        sb += handle(funcDef, CTX())
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

fun handle(funcDef: FuncdefContext, ctx: CTX): String {
  return if (ctx.isLambda)
    handle(funcDef.suite(), ctx)
  else {
    val sb = StringBuilder()
    sb += """fun ${funcDef.NAME().text}${funcDef.parameters().text}{
        ${handle(funcDef.suite(), ctx)}
      }
    """
    sb.toString()
  }
}

fun handle(suite: SuiteContext, ctx: CTX): String {
  val sb = StringBuilder()
  suite.simple_stmt()?.small_stmt()?.forEach {
    sb += handle(it, ctx)
  }
  suite.stmt()?.forEach {
    it?.simple_stmt()?.small_stmt()?.forEach {
      sb += handle(it, ctx)
    }
    it?.compound_stmt()?.let {
      sb += handle(it, ctx)
    }
  }
  return sb.toString()
}

fun handle(stmt: Small_stmtContext, ctx: CTX): String {
  val sb = StringBuilder()
  val st = stmt.getChild(0)
  when (st) {
    is Expr_stmtContext -> sb += handle(st, ctx)
    is Flow_stmtContext -> {
      val st = st.getChild(0)
      when (st) {
        is Return_stmtContext -> sb += handle(st, ctx)
        else -> sb += st.text + "\n"
      }
    }
    else -> {
      sb += """/* ignored
         ${st.text}
         */
         """
    }
  }
  return sb.toString()
}

fun handle(stmt: Compound_stmtContext, ctx: CTX): String = sb { out ->
  val st = stmt.getChild(0)
  
  when (st) {
    is If_stmtContext -> {
      val tests = st.test()
      val suites = st.suite()
      
      if (tests.size == 1) {
        out += """if(${handle(tests[0], ctx)}){
            ${handle(suites[0], ctx)}
          }
        """
        if (suites.size > tests.size) {
          //else branch
          out += """else{
              ${handle(suites.last(), ctx)}
            }
          """
        }
      } else {
        out += """when{
        """
        tests.zip(suites).forEach { (test, suite) ->
          out += """${handle(test, ctx)} -> {
              ${handle(suite, ctx)}
            }
          """
        }
        if (suites.size > tests.size) {
          //else branch
          out += """else -> {
              ${handle(suites.last(), ctx)}
            }
          """
        }
        out += "}\n"
      }
    }
    is For_stmtContext -> {
      out += st.text
    }
    is While_stmtContext -> {
      out += st.text
    }
    is With_stmtContext -> {
      out += st.with_item().joinToString(", ") {
        handle(it.test(), ctx) + (it.expr()?.let { " as " + handle(it, ctx) } ?: "")
      }
      out += "{\n"
      out += handle(st.suite(), ctx)
      out += "}\n"
    }
    is FuncdefContext -> {
      out += handle(st, ctx)
    }
    else -> {
      out += st.text + "\n"
    }
  }
}

fun handle(stmt: Return_stmtContext, ctx: CTX): String {
  val sb = StringBuilder()
  if (!ctx.isLambda)
    sb += "return "
  sb += handle(stmt.testlist(), ctx)
//  sb += "gradOutputs.append(${ctx.testlist().text})\n"
  return sb.toString()
}

fun handle(stmt: Expr_stmtContext, ctx: CTX): String = sb { out ->
  val star_expr = stmt.testlist_star_expr()
  val ann = stmt.annassign()
  val aug = stmt.augassign()
  
  when {
    ann != null -> {
      val tests = ann.test()
      out += "var ${handle(star_expr[0], ctx)}: ${tests[0]}"
      if (tests.size > 1)
        out += "=${handle(tests[1], ctx)}"
    }
    aug != null -> {
      val op = aug.getChild(0).text
      val right = handle(stmt.testlist(), ctx)
      val left = handle(star_expr[0], ctx)
      out += when (op) {
        "+=", "-=", "*=", "/=", "%=" -> "$left$op$right"
        "&=" -> "$left=$left and $right"
        "|=" -> "$left=$left or $right"
        "^=" -> "$left=$left xor $right"
        "<<=" -> "$left=$left shl $right"
        ">>=" -> "$left=$left shr $right"
        "**=" -> "$left=pow($left,$right)"
        "//=" -> "$left=($left / $right).toInt()"
        else -> error("$op")
      }
    }
    else -> {
      if (star_expr.size == 1)
        out += handle(star_expr[0], ctx)
      else {
        val tests = star_expr[0].test()
        if (tests != null && tests.size == 1) {
          val v0 = handle(tests[0], ctx)
          if (v0 !in ctx.vars) {
            out += "var "
            ctx.vars += v0
          }
          out += v0
        } else {
          out += "var " + handle(star_expr[0], ctx)
        }
        for (s in star_expr.drop(1))
          out += "=" + handle(s, ctx)
        
      }
      out += "\n"
    }
  }
}

fun handle(testlist_star: Testlist_star_exprContext, ctx: CTX): String = sb { out ->
  testlist_star.test()?.let {
    out += it.joinToString(", ") { handle(it, ctx) }
  }
  testlist_star.star_expr()?.let {
    out += it.joinToString(", ") { handle(it, ctx) }
  }
}