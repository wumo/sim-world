package wumo.sim.util

fun String.print(suffix: String) = kotlin.io.print(this + suffix)
fun String.println(suffix: String) = kotlin.io.println(this + suffix)
fun String.println() = kotlin.io.println(this)