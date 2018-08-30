package wumo.sim.tensorflow.gen.python

//
//fun handle(stmt: Compound_stmtContext, ctx: CTX): String = sb { out ->
//  val st = stmt.getChild(0)
//
//  out += when (st) {
//    is If_stmtContext -> {
//      handle(st, ctx)
//    }
//    is For_stmtContext -> {
//      handle(st, ctx)
//    }
//    is While_stmtContext -> {
//      handle(st, ctx)
//    }
//    is With_stmtContext -> {
//      handle(st, ctx)
//    }
//    is FuncdefContext -> {
//      handle(st, ctx)
//    }
//    else -> {
//      st.text + "\n"
//    }
//  }
//}
//
//fun handle(st: If_stmtContext, ctx: CTX): String = sb { out ->
//  val tests = st.test()
//  val suites = st.suite()
//
//  if (tests.size == 1) {
//    out += """if(${handle(tests[0], ctx)}){
//            ${handle(suites[0], ctx)}
//          }
//        """
//    if (suites.size > tests.size) {
//      //else branch
//      out += """else{
//              ${handle(suites.last(), ctx)}
//            }
//          """
//    }
//  } else {
//    out += """when{
//        """
//    tests.zip(suites).forEach { (test, suite) ->
//      out += """${handle(test, ctx)} -> {
//              ${handle(suite, ctx)}
//            }
//          """
//    }
//    if (suites.size > tests.size) {
//      //else branch
//      out += """else -> {
//              ${handle(suites.last(), ctx)}
//            }
//          """
//    }
//    out += "}\n"
//  }
//}
//
//fun handle(st: While_stmtContext, ctx: CTX): String = sb { out ->
//  val test = st.test()
//  val suites = st.suite()
//  out += """while(${handle(test, ctx)}){
//            ${handle(suites[0], ctx)}
//          }
//    ${if (suites.size > 1)
//    handle(suites[1], ctx)
//  else ""
//  }
//
//        """
//}
//
//fun handle(st: For_stmtContext, ctx: CTX): String = sb { out ->
//  val exprList = st.exprlist()
//  val testList = st.testlist()
//  val suites = st.suite()
//  out += """for(${handle(exprList, ctx)} in ${handle(testList, ctx)}){
//            ${handle(suites[0], ctx)}
//          }
//    ${if (suites.size > 1)
//    handle(suites[1], ctx)
//  else ""
//  }
//        """
//}
//
//fun handle(st: With_stmtContext, ctx: CTX): String = sb { out ->
//  out += st.with_item().joinToString(", ") {
//    handle(it.test(), ctx) + (it.expr()?.let { " as " + handle(it, ctx) } ?: "")
//  }
//  out += "{\n"
//  out += handle(st.suite(), ctx)
//  out += "}\n"
//}
//
//fun handle(st: DecoratedContext, ctx: CTX): String = sb { out ->
//  val opTypes = st.decorators().decorator().filter { it.dotted_name().text == "ops.RegisterGradient" }
//      .map {
//        val args = it.arglist().argument()
//        assert(args.size == 1)
//        args[0].text
//      }
//
//  if (opTypes.isEmpty()) return@sb
//  out += """register(${opTypes.joinToString(", ")}){op,grad->
//              ${handle(st.funcdef(), CTX(isRegister = true))}
//            }
//            """
//}
