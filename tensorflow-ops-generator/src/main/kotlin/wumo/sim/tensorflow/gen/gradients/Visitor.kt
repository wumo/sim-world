package wumo.sim.tensorflow.gen.gradients

import org.antlr.v4.runtime.tree.ErrorNode
import org.antlr.v4.runtime.tree.TerminalNode
import wumo.python3.Python3BaseVisitor
import wumo.python3.Python3Parser.*
import wumo.sim.tensorflow.gen.toCamelCase
import wumo.sim.util.sb

open class Context(var outer: Context? = null) {
  var isRegister: Boolean = false
  val vars: MutableSet<String> = mutableSetOf()
  var functionName: String = ""
  
  init {
    outer?.let {
      vars += it.vars
      functionName = it.functionName
      isRegister = it.isRegister
    }
  }
}

var currentContext = Context()

fun <R> context(init: Context.() -> Unit = {}, block: () -> R): R {
  val parent = currentContext
  currentContext = Context(parent)
  init(currentContext)
  try {
    return block()
  } finally {
    currentContext = parent
  }
}

class Visitor(val name: String) : Python3BaseVisitor<String>() {
  override fun visitErrorNode(node: ErrorNode): String {
    return node.text
  }
  
  override fun visitTerminal(node: TerminalNode): String {
    return when (node.symbol.type) {
      NEWLINE -> ""
      NAME -> node.text.toCamelCase()
      else -> node.text
    }
  }
  
  override fun defaultResult(): String = ""
  
  override fun aggregateResult(aggregate: String, nextResult: String): String {
    return aggregate + nextResult
  }
  
  override fun visitFile_input(ctx: File_inputContext) = sb {
    +"""
    import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
    import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
    import wumo.sim.util.append
    import wumo.sim.tensorflow.tf
    fun register_$name(){
    """
    ctx.stmt()?.forEach {
      +visit(it)
    }
    +"}"
  }
  
  override fun visitDecorated(ctx: DecoratedContext) = sb {
    val opTypes = ctx.decorators().decorator()
        .filter { it.dotted_name().text == "ops.RegisterGradient" }
        .map {
          val args = it.arglist().argument()
          assert(args.size == 1)
          args[0].text
        }
    if (opTypes.isEmpty()) return@sb
    +"register(${opTypes.joinToString(", ")}){op,grad->\nval grad = grad[0]!!.toOutput()\n"
    context({ isRegister = true }) {
      +visit(ctx.funcdef())
    }
    +"}\n"
  }
  
  override fun visitImport_stmt(ctx: Import_stmtContext): String = "/* ${ctx.text} */\n"
  override fun visitDel_stmt(ctx: Del_stmtContext): String = "/* ${ctx.text} */\n"
  override fun visitPass_stmt(ctx: Pass_stmtContext): String = "/* ${ctx.text} */\n"
  
  override fun visitFuncdef(ctx: FuncdefContext) = sb {
    context {
      currentContext.functionName =
          if (currentContext.isRegister) "register"
          else ctx.NAME().text
      ctx.parameters().typedargslist().tfpdef()?.let {
        for (tfpdef in it)
          currentContext.vars += tfpdef.NAME().text
      }
      if (currentContext.isRegister)
        +visit(ctx.suite())
      else {
        +"fun ${ctx.NAME().text.toCamelCase()}${ctx.parameters().text.toCamelCase()}{"
        +visit(ctx.suite())
        +"}\n"
      }
    }
  }
  
  override fun visitExpr_stmt(stmt: Expr_stmtContext) = sb {
    val star_expr = stmt.testlist_star_expr()
    val ann = stmt.annassign()
    val aug = stmt.augassign()
    
    when {
      ann != null -> {
        val tests = ann.test()
        +"val ${visit(star_expr[0])}: ${tests[0]}"
        if (tests.size > 1)
          +"=${visit(tests[1])}"
      }
      aug != null -> {
        val op = aug.getChild(0).text
        val right = visit(stmt.testlist())
        val left = visit(star_expr[0])
        +when (op) {
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
          +visit(star_expr[0])
        else {
          val tests = star_expr[0].test()
          if (tests != null && tests.size == 1) {
            val v0 = visit(tests[0])
            if (v0 !in currentContext.vars) {
              +"val "
              currentContext.vars += v0
            }
            +v0
          } else
            +"val (${visit(star_expr[0])})"
          for (s in star_expr.drop(1))
            +"=${visit(s)}"
        }
        +"\n"
      }
    }
  }
  
  override fun visitBreak_stmt(ctx: Break_stmtContext) = "break\n"
  override fun visitContinue_stmt(ctx: Continue_stmtContext) = "continue\n"
  override fun visitYield_stmt(ctx: Yield_stmtContext) = "yield\n"
  override fun visitRaise_stmt(ctx: Raise_stmtContext) = "${ctx.text}\n"
  override fun visitAssert_stmt(ctx: Assert_stmtContext) = "${ctx.text}\n"
  override fun visitReturn_stmt(ctx: Return_stmtContext) = sb {
    if (currentContext.isRegister) {
      ctx.testlist()?.let {
        val text = visit(it)
        if (it.test().size == 1 && text.startsWith("listOf("))
          +text + " //return@register"
        else
          +"listOf(${visit(it)})  //return@register"
      }
    } else {
      +"return ${ctx.testlist()?.let { visit(it) }}"
    }
  }
  
  override fun visitIf_stmt(ctx: If_stmtContext) = sb {
    
    val tests = ctx.test()
    val suites = ctx.suite()
    
    if (tests.size == 1) {
      context {
        +"""if(${visit(tests[0])}){
              ${visit(suites[0])}
            }
        """
      }
      context {
        if (suites.size == tests.size + 1) //else branch
          +"""else{
              ${visit(suites.last())}
          }
          """
      }
    } else {
      +"""when{
        """
      tests.zip(suites).forEach { (test, suite) ->
        context {
          +"""${visit(test)} -> {
              ${visit(suite)}
            }
          """
        }
      }
      if (suites.size == tests.size + 1) //else branch
        context {
          +"""else -> {
              ${visit(suites.last())}
            }
          """
        }
      +"}\n"
    }
  }
  
  override fun visitWhile_stmt(ctx: While_stmtContext) = sb {
    context {
      val test = ctx.test()
      val suites = ctx.suite()
      +"""while(${visit(test)}){
            ${visit(suites[0])}
          }
      """
      if (suites.size > 1)
        +visit(suites[1])
    }
  }
  
  override fun visitFor_stmt(ctx: For_stmtContext) = sb {
    context {
      val exprList = ctx.exprlist()
      val testList = ctx.testlist()
      val suites = ctx.suite()
      +"""
        for(${visit(exprList)} in ${visit(testList)}){
          ${visit(suites[0])}
        }
      """
      if (suites.size > 1)
        +visit(suites[1])
    }
  }
  
  override fun visitTry_stmt(ctx: Try_stmtContext) = sb {
    context {
      +"""
        try{
          ${visit(ctx.try_suite())}
          //else if no exception
          ${ctx.else_suite()?.let { visit(it) }}
        }
      """
      val except = ctx.except_clause()
      if (except != null) {
        val except_suites = ctx.except_suite()
        var i = 0
        for (e in except)
          +"""catch(_:${e.test()?.let { visit(it) } ?: "Exception"}){
            ${visit(except_suites[i++])}
          }
        """
      }
      ctx.finally_suite()?.let {
        +"""
        finally{
          ${visit(it)}
        }
      """
      }
    }
  }
  
  override fun visitWith_stmt(ctx: With_stmtContext) = sb {
    context {
      val items = ctx.with_item()
      for (item in items)
        +visit(item.test())
      +"""{
        ${visit(ctx.suite())}
      }
      """
    }
  }
  
  override fun visitOr_test(ctx: Or_testContext): String =
      ctx.and_test().joinToString(" || ") { visit(it) }
  
  override fun visitAnd_test(ctx: And_testContext): String =
      ctx.not_test().joinToString(" && ") { visit(it) }
  
  override fun visitNot_test(ctx: Not_testContext): String =
      (ctx.not_test()?.let {
        "!" + visit(it)
      } ?: ctx.comparison()?.let {
        visit(it)
      })!!
  
  override fun visitComp_op(ctx: Comp_opContext): String =
      " " + when (ctx.text) {
        "is" -> "=="
        "<>", "isnot" -> "!="
        "notin" -> "!in"
        else -> ctx.text
      } + " "
  
  override fun visitExpr(ctx: ExprContext): String =
      ctx.xor_expr().joinToString(" or ") { visit(it) }
  
  override fun visitXor_expr(ctx: Xor_exprContext): String =
      ctx.and_expr().joinToString(" xor ") { visit(it) }
  
  override fun visitAnd_expr(ctx: And_exprContext): String =
      ctx.shift_expr().joinToString(" and ") { visit(it) }
  
  override fun visitShift_expr(ctx: Shift_exprContext): String =
      ctx.children.joinToString(" ") {
        var result = ""
        when (it) {
          is Arith_exprContext -> result = visit(it)
          is TerminalNode ->
            when (it.symbol.type) {
              LEFT_SHIFT -> result = "shl"
              RIGHT_SHIFT -> result = "shr"
            }
        }
        result
      }
  
  override fun visitArith_expr(ctx: Arith_exprContext): String =
      ctx.children.joinToString(" ") {
        var result = ""
        when (it) {
          is TermContext -> result = visit(it)
          is TerminalNode -> result = it.text
        }
        result
      }
  
  override fun visitTerm(ctx: TermContext): String =
      ctx.children.joinToString(" ") {
        var result = ""
        when (it) {
          is FactorContext -> result = visit(it)
          is TerminalNode ->
            result = when (it.symbol.type) {
              STAR, DIV, MOD -> it.text
              IDIV -> "//"
              else -> error("not supported${it.text}")
            }
        }
        result
      }
  
  override fun visitFactor(ctx: FactorContext): String = sb {
    ctx.factor()?.let {
      val op = ctx.getChild(0) as TerminalNode
      when (op.symbol.type) {
        ADD, MINUS -> +op + visit(it)
        NOT_OP -> +"(${visit(it)}).inv()"
      }
      Unit
    }
    ctx.power()?.let {
      +visit(it)
    }
  }
  
  override fun visitPower(ctx: PowerContext): String =
      if (ctx.factor() != null)
        "pow(${visit(ctx.atom_expr())},${visit(ctx.factor())})"
      else
        visit(ctx.atom_expr())
  
  override fun visitAtom_expr(ctx: Atom_exprContext): String = sb {
    ctx.AWAIT()?.let { +"await " }
    val atom = ctx.atom()
    val trailers = ctx.trailer()
    val c = atom.children[0] as TerminalNode
    when (c.symbol.type) {
      NAME -> {
        if (trailers.isNotEmpty()) {
          val dottedName = mutableListOf(c.text)
          var i = 0
          outer@ for (trailer in trailers) {
            val t = trailer.children[0] as TerminalNode
            when (t.symbol.type) {
              DOT -> dottedName += trailer.NAME().text
              else -> break@outer
            }
            i++
          }
          if (i < trailers.size) {
            val trailer = trailers[i]
            val t = trailer.children[0] as TerminalNode
            if (t.symbol.type == OPEN_PAREN) {//function call
              +functionCallReplacement(dottedName, trailer)
              i++
            } else
              +dottedName.joinToString(".") { it.toCamelCase() }
            for (j in i..trailers.lastIndex) {
              val trailer = trailers[j]
              trailer.NAME()?.let {
                when (it.text) {
                  "ndtype" -> +".dataType"
                  else -> +".${it.text}"
                }
              } ?: +visit(trailer)
            }
          } else {//only dotted name
            var text = dottedName.joinToString(".") { it.toCamelCase() }
            text = if (text.startsWith("ops.dtypes.") ||
                text.startsWith("dtypes.")) {
              val i = text.lastIndexOf('.')
              text.substring(i + 1).toUpperCase()
            } else {
              text.replace(".ndtype", ".dataType")
            }
            +text
          }
        } else
          +c.text.toCamelCase()
      }
      OPEN_PAREN -> {//list
        +"("
        atom.yield_expr()?.let { +visit(it) }
        atom.testlist_comp()?.let { +visit(it) }
        +")"
        trailers?.forEach { +visit(it) }
      }
      OPEN_BRACK -> {//array
        +"listOf("
        atom.testlist_comp()?.let { +visit(it) }
        +")"
        trailers?.forEach { +visit(it) }
      }
      OPEN_BRACE -> {//dictionary
        +"mapOf("
        atom.dictorsetmaker()?.let { +visit(it) }
        +")"
        trailers?.forEach { +visit(it) }
      }
      ELLIPSIS -> {
        +"..."
        trailers?.forEach { +visit(it) }
      }
      NUMBER -> {
        +c.text
        trailers?.forEach { +visit(it) }
      }
      STRING -> atom.STRING().forEach {
        var text = it.text
        if (text.startsWith("\"\"\"") &&
            text.endsWith("\"\"\""))
          text = "/**" + text.substring(3, text.length - 3) + "*/"
        +text
        trailers?.forEach { +visit(it) }
      }
      NONE -> {
        +" null "
        trailers?.forEach { +visit(it) }
      }
      TRUE -> {
        +" true "
        trailers?.forEach { +visit(it) }
      }
      FALSE -> {
        +" false"
        trailers?.forEach { +visit(it) }
      }
    }
  }
  
  fun functionCallReplacement(dottedName: List<String>,
                              trailer: TrailerContext) = sb {
    val argString = trailer.arglist()?.let {
      visit(it)
    } ?: ""
    when {
      dottedName.size == 1 -> {
        val functionName = dottedName[0]
        when (functionName) {
          "isinstance" -> {
            val argments = trailer.arglist().argument()
            +"(" + visit(argments[0]) + " is " + visit(argments[1]) + ")"
          }
          "len" -> {
            +"(" + argString + ").size"
          }
          else -> +functionName.toCamelCase() + "($argString)"
        }
      }
      dottedName.size == 2 -> {
        val packageName = dottedName[0]
        val functionName = dottedName[1]
        when (packageName) {
          "tensor_util", "constant_value", "array_ops",
          "math_ops", "constant_op", "control_flow_ops",
          "data_flow_ops", "nn_ops", "random_ops",
          "sparse_ops", "state_ops", "manip_ops",
          "tensor_array_ops"-> {
            val fn = when (functionName) {
              "constant" -> "const"
              "multiply" -> "mul"
              "subtract" -> "sub"
              "reduce_sum" -> "sum"
              else -> functionName
            }
            +"tf.${fn.toCamelCase()}($argString)"
          }
          "ops" -> {
            when (functionName) {
              "RegisterGradient" ->
                +"register($argString)"
              "NotDifferentiable" ->
                +"registerNonDifferentiable($argString)"
              else ->
                +"tf.${functionName.toCamelCase()}($argString)"
            }
          }
          else -> {
            if (packageName.startsWith("gen_") &&
                packageName.endsWith("_ops")) {
              +"tf.${functionName.toCamelCase()}($argString)"
            } else
              +"${packageName.toCamelCase()}.${functionName.toCamelCase()}($argString)"
          }
        }
      }
      else -> +dottedName.joinToString(".") { it.toCamelCase() } + "($argString)"
    }
  }
}