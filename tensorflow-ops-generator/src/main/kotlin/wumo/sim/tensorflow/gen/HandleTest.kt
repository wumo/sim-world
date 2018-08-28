package wumo.sim.tensorflow.gen

import org.antlr.v4.runtime.tree.TerminalNode
import wumo.python3.Python3Parser.*
import wumo.sim.util.plusAssign
import wumo.sim.util.sb

fun handle(testList: TestlistContext, ctx: CTX): String = sb {
  val tests = testList.test()
  it += tests.joinToString(", ") { handle(it, ctx) }
}

fun handle(test: TestContext, ctx: CTX): String = sb {
  val or_test = test.or_test()
  assert(or_test.size == 1)
  it += handle(or_test[0], ctx)
}

fun handle(or_test: Or_testContext, ctx: CTX): String = sb {
  val and_tests = or_test.and_test()
  it += and_tests.joinToString(" || ") { handle(it, ctx) }
}

fun handle(and_test: And_testContext, ctx: CTX): String = sb {
  val not_tests = and_test.not_test()
  it += not_tests.joinToString(" && ") { handle(it, ctx) }
}

fun handle(not_test: Not_testContext, ctx: CTX): String = sb { out ->
  not_test.not_test()?.let {
    out += "!" + handle(it, ctx)
  }
  not_test.comparison()?.let {
    out += handle(it, ctx)
  }
}

fun handle(comparison: ComparisonContext, ctx: CTX): String = sb {
  val exprs = comparison.expr()
  it += handle(exprs[0], ctx)
  for ((comp_op, expr) in comparison.comp_op().zip(exprs.drop(1))) {
    it += when (comp_op.text) {
      "<", ">", "==", ">=", "<=", "!=", "in" -> comp_op.text
      "is" -> "=="
      "<>", "isnot" -> "!="
      "notin" -> "!in"
      else -> comp_op.text
    }
    it += " " + handle(expr, ctx)
  }
}

fun handle(expr: ExprContext, ctx: CTX): String = sb {
  it += expr.xor_expr().joinToString(" or ") { handle(it, ctx) }
}

fun handle(xor_expr: Xor_exprContext, ctx: CTX): String = sb {
  it += xor_expr.and_expr().joinToString(" xor ") { handle(it, ctx) }
}

fun handle(and_expr: And_exprContext, ctx: CTX): String = sb {
  it += and_expr.shift_expr().joinToString(" and ") { handle(it, ctx) }
}

fun handle(shift_expr: Shift_exprContext, ctx: CTX): String = sb {
  val ariths = shift_expr.arith_expr()
  it += handle(ariths[0], ctx)
  val rest = ariths.drop(1)
  for ((op, arith) in List(rest.size) { shift_expr.getChild(it * 2 + 1) }.zip(rest)) {
    when (op.text) {
      "<<" -> it += " shl " + handle(arith, ctx)
      ">>" -> it += " shr " + handle(arith, ctx)
    }
  }
}

fun handle(arith_expr: Arith_exprContext, ctx: CTX): String = sb {
  val terms = arith_expr.term()
  it += handle(terms[0], ctx)
  val rest = terms.drop(1)
  for ((op, term) in List(rest.size) { arith_expr.getChild(2 * it + 1) }.zip(rest)) {
    it += op.text + handle(term, ctx)
  }
}

fun handle(term: TermContext, ctx: CTX): String = sb { out ->
  val factors = term.factor()
  out += handle(factors[0], ctx)
  val rest = factors.drop(1)
  for ((op, factor) in List(rest.size) { term.getChild(2 * it + 1) }.zip(rest)) {
    val op = op.text
    out += when (op) {
      "*", "/", "%" -> op
      "//" -> "//"
      else -> error("not supported$op")
    }
    out += handle(factor, ctx)
  }
}

fun handle(factor: FactorContext, ctx: CTX): String = sb { out ->
  factor.factor()?.let {
    val op = factor.getChild(0).text
    when (op) {
      "+", "-" -> out += op + handle(it, ctx)
      "~" -> out += "(" + handle(it, ctx) + ").inv()"
    }
  }
  factor.power()?.let {
    out += handle(it, ctx)
  }
}

fun handle(power: PowerContext, ctx: CTX): String = sb { out ->
  val factor = power.factor()
  out += if (factor != null) {
    "pow(${handle(power.atom_expr(), ctx)},${handle(power.factor(), ctx)})"
  } else
    handle(power.atom_expr(), ctx)
}

fun handle(atom_expr: Atom_exprContext, ctx: CTX): String = sb { out ->
  atom_expr.AWAIT()?.let {
    out += "AWAIT "
  }
  out += handle(atom_expr.atom(), ctx)
  val trailers = atom_expr.trailer()
  trailers?.let {
    for (trailer in it) {
      out += handle(trailer, ctx)
    }
  }
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
    "dtype" to "dataType")

fun process(name: String): String = sb {
  val cs = name.toCharArray()
  for ((i, c) in cs.withIndex()) {
    if (c == '_' && i + 1 < cs.size)
      cs[i + 1] = cs[i + 1].toUpperCase()
    else
      it += c
  }
}

fun handle(atom: AtomContext, ctx: CTX): String = sb { out ->
  val c = atom.getChild(0) as TerminalNode
  when (c.symbol.type) {
    OPEN_PAREN, OPEN_BRACK -> {
      val list = atom.testlist_comp()
      out += handle(list, ctx)
    }
    OPEN_BRACE -> {
      atom.dictorsetmaker()
    }
    NAME -> {
      var name = atom.NAME().text
      name = nameReplace[name] ?: name
      name = process(name)
      out += name
    }
    NUMBER -> {
      val number = atom.NUMBER()
      out += number.text
    }
    STRING -> {
      val strs = atom.STRING()
      for (str in strs) {
        out += str
      }
    }
    ELLIPSIS -> {
      out += "..."
    }
    NONE -> {
      out += " null "
    }
    TRUE -> {
      out += " true "
    }
    FALSE -> {
      out += " false "
    }
    else -> {
      error("$c")
    }
  }
}

fun handle(trailer: TrailerContext, ctx: CTX): String = sb { out ->
  trailer.arglist()?.let {
    out += "(" + handle(it, ctx) + ")"
  }
  trailer.subscriptlist()?.let {
    out += "[" + handle(it, ctx) + "]"
  }
  trailer.NAME()?.let {
    var name = it.text
    name = nameReplace[name] ?: name
    name = process(name)
    out += "." + name
  }
}

fun handle(arglist: ArglistContext, ctx: CTX): String = sb {
  it += arglist.argument().joinToString(", ") { handle(it, ctx) }
}

fun handle(argument: ArgumentContext, ctx: CTX): String = sb { out ->
  val tests = argument.test()
  val c = argument.getChild(0)
  if (c is TerminalNode) {
    when (c.symbol.type) {
      STAR -> {
        out += "*" + handle(tests[0], ctx)
      }
      POWER -> {
        out += "**" + handle(tests[0], ctx)
      }
    }
  } else
    when {
      tests.size == 1 -> {
        val comp_for = argument.comp_for()
        out += if (comp_for != null) {
          handle(tests[0], ctx) + comp_for.text
        } else {
          handle(tests[0], ctx)
        }
      }
      tests.size > 1 -> {
        for ((name, arg) in tests.zipWithNext()) {
          out += handle(name, ctx) + "=" + handle(arg, ctx)
        }
      }
      else -> {
      
      }
    }
}

fun handle(subscript: SubscriptlistContext, ctx: CTX): String = sb {
  it += subscript.text
}

fun handle(testList: Testlist_compContext, ctx: CTX): String = sb { out ->
  val comp_for = testList.comp_for()
  if (comp_for != null) {//for list
    out += testList.text
  } else {
    out += "listOf("
    testList.test()?.let {
      out += it.joinToString(", ") { handle(it, ctx) }
    }
    testList.star_expr()?.let {
      out += it.joinToString(", ") { handle(it, ctx) }
    }
    out += ")"
  }
  
}

fun handle(starExpr: Star_exprContext, ctx: CTX): String = sb {
  it += "*${handle(starExpr.expr(), ctx)}"
}