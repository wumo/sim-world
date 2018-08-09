package wumo.sim.tensorflow.learn_lowlevel_api

import okio.BufferedSink
import wumo.sim.util.readString
import wumo.sim.util.sink
import java.io.File
import java.nio.charset.Charset

object RetrieveOpGroups {
  @JvmStatic
  fun main(args: Array<String>) {
    val opNamePattern = Regex("""tensorflow::NodeBuilder\s*[(]\s*[^)]+?\s*,\s*"(\w+)"\s*[)]""")
    val excludes = setOf("decode_proto_ops",
                         "encode_proto_ops", "remote_fused_graph_ops",
                         "rpc_ops", "sendrecv_ops")
    File("algorithm/src/test/kotlin/wumo/sim/algorithm/tensorflow/learn_lowlevel_api/ops.kt").sink { out ->
      
      val opGroups = linkedMapOf<String, MutableSet<String>>()
      gather(excludes, opNamePattern, opGroups)
      dump(opGroups, out)
    }
  }
  
  fun gather(excludes: Set<String>, opNamePattern: Regex, opGroups: LinkedHashMap<String, MutableSet<String>>) {
    File("algorithm/resources/ops").listFiles().forEach {
      val opGroupName = it.nameWithoutExtension
      for (exclude in excludes) if (opGroupName.startsWith(exclude)) return@forEach
      var result = opNamePattern.find(readString(it)) ?: return@forEach
      val ops = linkedSetOf<String>()
      do {
        val opName = result.groupValues[1]
        ops += opName
        val tmp = result.next() ?: break
        result = tmp
      } while (true)
      val groupName = opGroupName.replace("_internal", "")
      opGroups.compute(groupName) { _, v ->
        if (v != null) {
          v.addAll(ops)
          v
        } else
          ops
      }
    }
  }
  
  fun dump(opGroups: LinkedHashMap<String, MutableSet<String>>, out: BufferedSink): BufferedSink? {
    val sb = StringBuilder()
    var firstFile = true
    sb.append("""
          |import wumo.sim.util.a
          |
          |val opGroups = mapOf(
          |""".trimMargin())
    for ((groupName, ops) in opGroups) {
      if (!firstFile) sb.append(",\n") else firstFile = false
      sb.append("\"$groupName\" to a(")
      var firstOp = true
      for ((i, op) in ops.withIndex()) {
        if (!firstOp) sb.append(',') else firstOp = false
        if (i % 11 == 0) sb.append("\n    ")
        sb.append("\"$op\"")
      }
      sb.append(')')
    }
    sb.append(")")
    return out.writeString(sb.toString(), Charset.defaultCharset())
  }
}