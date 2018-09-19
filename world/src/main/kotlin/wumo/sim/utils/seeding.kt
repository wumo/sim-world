package wumo.sim.utils

import wumo.sim.util.errorIf
import wumo.sim.util.ndarray.abs
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.abs
import kotlin.random.Random

fun np_random(seed: Long? = null): Pair<Random, Long> {
  errorIf(seed != null && seed < 0) {
    "Seed must be a non-negative integer or omitted, not $seed"
  }
  val seed = create_seed(seed)
  val rng = Random(hash_seed(seed))
  return rng to seed
}

fun create_seed(seed: Long? = null): Long =
    when (seed) {
      null -> ByteBuffer.wrap(SecureRandom().generateSeed(Long.SIZE_BYTES)).long
      else -> seed % (1 shl (8 * Long.SIZE_BYTES))
    }

fun hash_seed(seed: Long? = null): Long {
  val seed = seed ?: create_seed()
  val hash = MessageDigest.getInstance("SHA-512")
      .digest(seed.toString().toByteArray())
  return ByteBuffer.wrap(
      hash.sliceArray(0 until Long.SIZE_BYTES)).long
}
