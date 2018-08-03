package wumo.sim.util

import java.io.File
import java.nio.charset.Charset

fun readAll(file: String): ByteArray {
  File(file).source().buffer().use {
    return it.readByteArray()
  }
}

fun dump(file: String, data: String): String {
  File(file).sink().buffer().use {
    it.writeString(data, Charset.defaultCharset())
    return data
  }
}

fun dump(file: String, data: ByteArray) {
  File(file).sink().buffer().use {
    it.write(data)
  }
}