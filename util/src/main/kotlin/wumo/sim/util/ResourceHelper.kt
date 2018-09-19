package wumo.sim.util

import java.io.File
import java.io.File.separatorChar
import java.nio.file.Files.copy
import java.nio.file.StandardCopyOption

fun unpackFileToTemp(resource: String, override: Boolean = false): String {
  val file = File(System.getProperty("java.io.tmpdir") +
                      separatorChar + resource)
  if (!override && file.exists()) return file.path
  file.parentFile.mkdirs()
  val source = Thread.currentThread().contextClassLoader
      .getResourceAsStream(resource)
  copy(source, file.toPath(), StandardCopyOption.REPLACE_EXISTING)
  return file.path
}

fun unpackDirToTemp(resourceDir: String, override: Boolean = false): String {
  val loader = Thread.currentThread().contextClassLoader
  val dir = File(loader.getResource(resourceDir).path)
  dir.listFiles().forEach {
    unpackFileToTemp(resourceDir + separatorChar + it.name,
                     override)
  }
  return System.getProperty("java.io.tmpdir") + separatorChar + resourceDir
}

fun listResources(resourceDir: String): Array<File> {
  val loader = Thread.currentThread().contextClassLoader
  val dir = File(loader.getResource(resourceDir).path)
  return dir.listFiles()
}