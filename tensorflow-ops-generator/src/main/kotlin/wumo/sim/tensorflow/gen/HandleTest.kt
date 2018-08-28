package wumo.sim.tensorflow.gen

import wumo.python3.Python3Parser
import wumo.python3.Python3Parser.*
import wumo.sim.util.plusAssign
import wumo.sim.util.sb

fun handle(testList: TestlistContext): String = sb {
  val tests = testList.test()
  it += tests.joinToString(", ") { handle(it) }
}

fun handle(test: Python3Parser.TestContext): String = sb {
  val or_test = test.or_test()
  assert(or_test.size == 1)
  it += handle(or_test[0])
}

fun handle(or_test: Or_testContext): String = sb {
  val and_tests = or_test.and_test()
  it += and_tests.joinToString(" || ") { handle(it) }
}

fun handle(and_test: And_testContext): String = sb {
  val not_tests = and_test.not_test()
  it += not_tests.joinToString(" && ") { handle(it) }
}

fun handle(not_test: Not_testContext): String = sb { out ->
  not_test.not_test()?.let {
    out += "!" + handle(it)
  }
  not_test.comparison()?.let {
    out += handle(it)
  }
}

fun handle(comparison: ComparisonContext): String = sb {
  val exprs = comparison.expr()
  it += handle(exprs[0])
  for ((comp_op, expr) in comparison.comp_op().zip(exprs.drop(1))) {
    it += when (comp_op.text) {
      "<", ">", "==", ">=", "<=", "!=", "in" -> comp_op.text
      "<>" -> "!="
      "not in" -> "!in"
      else -> comp_op.text
    }
    it += handle(expr)
  }
}

fun handle(expr: ExprContext): String = sb {
  it += expr.xor_expr().joinToString(" or ") { handle(it) }
}

fun handle(xor_expr: Xor_exprContext): String = sb {
  it += xor_expr.and_expr().joinToString(" xor ") { handle(it) }
}

fun handle(and_expr: And_exprContext): String = sb {
  it += and_expr.shift_expr().joinToString(" and ") { handle(it) }
}

fun handle(shift_expr: Shift_exprContext): String = sb {
  val ariths = shift_expr.arith_expr()
  it += handle(ariths[0])
  val rest = ariths.drop(1)
  for ((op, arith) in List(rest.size) { shift_expr.getChild(it * 2 + 1) }.zip(rest)) {
    when (op.text) {
      "<<" -> it += " shl " + handle(arith)
      ">>" -> it += " shr " + handle(arith)
    }
  }
}

fun handle(arith_expr: Arith_exprContext): String = sb {
  val terms = arith_expr.term()
  it += handle(terms[0])
  val rest = terms.drop(1)
  for ((sign, term) in List(rest.size) { arith_expr.getChild(2 * it + 1) }.zip(rest)) {
    it += sign.text + handle(term)
  }
}

fun handle(term: TermContext): String = sb {
  val factors = term.factor()
  it += handle(factors[0])
  val rest = factors.drop(1)
  for ((sign, factor) in List(rest.size) { term.getChild(2 * it + 1) }.zip(rest)) {
    it += sign.text + handle(factor)
  }
}

fun handle(factor: FactorContext): String = sb { out ->
  factor.factor()?.let {
    out += factor.getChild(0).text + handle(it)
  }
  factor.power()?.let {
    out += handle(it)
  }
}

fun handle(power: PowerContext): String = sb { out ->
  out += handle(power.atom_expr())
  power.factor()?.let {
    out += "**" + handle(power.factor())
  }
}

fun handle(atom_expr: Atom_exprContext): String = sb { out ->
  atom_expr.AWAIT()?.let {
    out += "AWAIT "
  }
  out += handle(atom_expr.atom())
  val trailers = atom_expr.trailer()
  trailers?.let {
    for (trailer in it) {
      out += handle(trailer)
    }
  }
}

fun handle(atom: AtomContext): String = sb { out ->
  atom.NAME()?.let {
    out += it.text
  }
  atom.NUMBER()?.let {
    out += it.text
  }
  atom.STRING()?.let {
    for (string in it) {
      out += string.text
    }
  }
  atom.dictorsetmaker()?.let {
    it
  }
  atom.yield_expr()?.let {
    it
  }
  atom.testlist_comp()?.let {
    it
  }
}

fun handle(trailer: TrailerContext): String = sb {

}
