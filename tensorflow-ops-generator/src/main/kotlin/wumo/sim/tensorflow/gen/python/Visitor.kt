package wumo.sim.tensorflow.gen.python

import org.antlr.v4.runtime.tree.ErrorNode
import org.antlr.v4.runtime.tree.TerminalNode
import wumo.python3.Python3BaseVisitor
import wumo.python3.Python3Parser.*
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
  
  val nameReplace = mapOf(
      "_shape_tuple" to "shape",
      "tensor_util" to "tf",
      "constant_value" to "const",
      "array_ops" to "tf",
      "math_ops" to "tf",
      "constant_op" to "tf",
      "constant" to "const",
      "gen_math_ops" to "tf",
      "gen_array_ops" to "tf",
      "dtype" to "dataType",
      "ops" to "tf")
  
  fun process(name: String): String = sb {
    val cs = name.toCharArray()
    for ((i, c) in cs.withIndex()) {
      if (c == '_' && i + 1 < cs.size)
        cs[i + 1] = cs[i + 1].toUpperCase()
      else
        +c
    }
  }
  
  override fun visitTerminal(node: TerminalNode): String {
    return when (node.symbol.type) {
      NEWLINE -> ""
      NAME -> {
        var name = node.text
        name = nameReplace[name] ?: name
        process(name)
      }
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
    +"register(${opTypes.joinToString(", ")}){op,grad->"
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
        +"fun ${ctx.NAME().text}${ctx.parameters().text}{"
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
        +"var ${visit(star_expr[0])}: ${tests[0]}"
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
              +"var "
              currentContext.vars += v0
            }
            +v0
          } else
            +"var ${visit(star_expr[0])}"
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
    +if (currentContext.isRegister)
      "return@${currentContext.functionName} ${ctx.testlist()?.let { visit(it) }}"
    else {
      "return ${ctx.testlist()?.let { visit(it) }}"
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
      +"""
        finally{
          ${visit(ctx.finally_suite())}
        }
      """
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
  
  override fun visitShift_expr(ctx: Shift_exprContext): String = sb {
    val ariths = ctx.arith_expr()
    +visit(ariths[0])
    val rest = ariths.drop(1)
    for ((op, arith) in List(rest.size) { ctx.getChild(it * 2 + 1) }.zip(rest))
      when (op.text) {
        "<<" -> +" shl ${visit(arith)}"
        ">>" -> +" shr ${visit(arith)}"
      }
  }
  
  override fun visitArith_expr(ctx: Arith_exprContext): String = sb {
    val terms = ctx.term()
    +visit(terms[0])
    val rest = terms.drop(1)
    for ((op, term) in List(rest.size) { ctx.getChild(2 * it + 1) }.zip(rest)) {
      +op.text + visit(term)
    }
  }
  
  override fun visitTerm(ctx: TermContext): String = sb { out ->
    val factors = ctx.factor()
    +visit(factors[0])
    val rest = factors.drop(1)
    for ((op, factor) in List(rest.size) { ctx.getChild(2 * it + 1) }.zip(rest)) {
      val op = op.text
      +when (op) {
        "*", "/", "%" -> op
        "//" -> "//"
        else -> error("not supported$op")
      }
      +visit(factor)
    }
  }
  
  override fun visitFactor(ctx: FactorContext): String = sb { out ->
    ctx.factor()?.let {
      val op = ctx.getChild(0).text
      when (op) {
        "+", "-" -> +op + visit(it)
        "~" -> +"(" + visit(it) + ").inv()"
      }
    }
    ctx.power()?.let {
      +visit(it)
    }
  }
  
  override fun visitPower(ctx: PowerContext): String {
    return super.visitPower(ctx)
  }
  
  override fun visitAtom(ctx: AtomContext): String {
    return super.visitAtom(ctx)
  }
}