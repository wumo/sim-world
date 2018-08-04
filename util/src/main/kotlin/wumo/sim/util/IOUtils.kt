package wumo.sim.util

import okio.BufferedSink
import okio.BufferedSource
import java.io.File
import java.nio.charset.Charset

inline fun <R> File.sink(block: (BufferedSink) -> R) =
    sink().buffer().use(block)

inline fun <R> File.source(block: (BufferedSource) -> R) =
    source().buffer().use(block)

fun readString(file: File): String {
  file.source().buffer().use {
    return it.readString(Charset.defaultCharset())
  }
}

fun readString(file: String): String {
  File(file).source().buffer().use {
    return it.readString(Charset.defaultCharset())
  }
}

fun readBytes(file: String): ByteArray {
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