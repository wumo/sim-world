package wumo.sim.util

import java.io.File
import java.nio.charset.Charset

fun dump(file: String, data: String): String {
  File(file).sink().buffer().use {
    it.writeString(data, Charset.defaultCharset())
    return data
  }
}