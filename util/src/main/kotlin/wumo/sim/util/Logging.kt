package wumo.sim.util

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import kotlin.reflect.KClass

fun <T : Any> logger(forClass: Class<T>): Logger {
  return LoggerFactory.getLogger(unwrapCompanionClass(forClass).name)
}

fun <T : Any> logger(forClass: KClass<T>): Logger {
  return LoggerFactory.getLogger(unwrapCompanionClass(forClass).simpleName)
}

// unwrap companion class to enclosing class given a Java Class
fun <T : Any> unwrapCompanionClass(ofClass: Class<T>): Class<*> {
  return ofClass.enclosingClass?.takeIf {
    it.kotlin.objectInstance?.javaClass == ofClass
  } ?: ofClass
}

// unwrap companion class to enclosing class given a Kotlin Class
fun <T : Any> unwrapCompanionClass(ofClass: KClass<T>): KClass<*> {
  return unwrapCompanionClass(ofClass.java).kotlin
}

// return a lazy logger property delegate for enclosing class
inline fun <reified R : Any> R.lazyLogger(): Lazy<Logger> {
  return lazy { logger(this.javaClass) }
}

inline fun Logger.warn(block: () -> String) {
  if (isWarnEnabled) warn(block())
}

inline fun Logger.info(block: () -> String) {
  if (isInfoEnabled) info(block())
}

inline fun Logger.debug(block: () -> String) {
  if (isDebugEnabled) debug(block())
}