package wumo.sim.algorithm.drl.common

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.ale.PLAYER_A_FIRE
import org.bytedeco.javacpp.ale.PLAYER_A_NOOP
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgproc.*
import wumo.sim.core.*
import wumo.sim.envs.atari.AtariEnv
import wumo.sim.envs.atari.AtariEnvType
import wumo.sim.envs.atari.AtariObsType
import wumo.sim.envs.envs
import wumo.sim.spaces.Box
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.*
import wumo.sim.util.ndarray.implementation.ByteArrayBuf
import wumo.sim.util.ndarray.implementation.FloatArrayBuf
import wumo.sim.util.ndarray.types.NDFloat
import wumo.sim.util.t4
import java.util.*
import kotlin.math.sign
import kotlin.random.Random
import kotlin.system.measureNanoTime
import kotlin.system.measureTimeMillis

fun make_atari(env_id: String): AtariEnvType {
  require("NoFrameskip" in env_id)
  var env = envs.Atari(env_id)
  env = NoopResetEnv(env, noop_max = 30)
  env = MaxAndSkipEnv(env, skip = 4)
  return env
}

fun wrap_atari_dqn(env: AtariEnvType) =
    wrap_deepmind(env, frame_stack = true, scale = true)

fun wrap_deepmind(env: AtariEnvType,
                  episode_life: Boolean = true,
                  clip_rewards: Boolean = true,
                  frame_stack: Boolean = false,
                  scale: Boolean = false): Env<NDArray<Float>, Float, Int, Int, Any> {
  var _env = env
  if (episode_life)
    _env = EpisodicLifeEnv(_env)
  if (PLAYER_A_FIRE in _env.unwrapped.action_set)
    _env = FireResetEnv(_env)
  _env = WarpFrame(_env)
  var env: Env<NDArray<Float>, Float, Int, Int, Any> = FloatFrame(_env)
  if (scale)
    env = ScaledFloatFrame(env)
  if (clip_rewards)
    env = ClipRewardEnv(env)
  if (frame_stack)
    env = FrameStack(env, 4)
  return env
}

class NoopResetEnv(env: AtariEnvType,
                   val noop_max: Int = 30)
  : Wrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  val noop_action: Int = 0
  var override_num_noops: Int? = null
  
  init {
    require(env.unwrapped.action_set[0] == PLAYER_A_NOOP)//NOOP
  }
  
  override fun reset(): AtariObsType {
    super.reset()
    val noops = override_num_noops ?: rand.nextInt(1, noop_max + 1)
    require(noops > 0)
    lateinit var obs: AtariObsType
    repeat(noops) {
      val (_obs, _, done) = env.step(noop_action)
      obs = _obs
      if (done)
        obs = env.reset()
    }
    return obs
  }
}

class MaxAndSkipEnv(env: AtariEnvType, val skip: Int = 4)
  : Wrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  val obs_buffer = mutableListOf(NDArray(env.observation_space.shape, 0.toByte()),
                                 NDArray(env.observation_space.shape, 0.toByte()))
  
  override fun step(a: Int): t4<AtariObsType, Float, Boolean, Map<String, Any>> {
    var total_reward = 0f
    var done = false
    lateinit var info: Map<String, Any>
    for (i in 0 until skip) {
      val (obs, reward, _done, _info) = env.step(a)
      done = _done
      info = _info
      if (i == skip - 2) obs_buffer[0] = obs
      if (i == skip - 1) obs_buffer[1] = obs
      total_reward += reward
      if (done) break
    }
    val a = obs_buffer[0]
    val b = obs_buffer[1]
    val c = ByteArray(a.size) {
      maxOf(a.rawGet(it), b.rawGet(it))
    }
    val max_frame = NDArray(a.shape, ByteArrayBuf(c), a.dtype)
//    val max_frame = obs_buffer.max(axis = 0)
    return t4(max_frame, total_reward, done, info)
  }
}

class EpisodicLifeEnv(env: AtariEnvType)
  : Wrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  var lives = 0
  var was_real_done = true
  
  override fun step(a: Int): t4<AtariObsType, Float, Boolean, Map<String, Any>> {
    val result = env.step(a)
    var done = result._3
    was_real_done = done
    val lives = env.unwrapped.ale.lives()
    if (lives < this.lives && lives > 0)
      done = true
    this.lives = lives
    result._3 = done
    return result
  }
  
  override fun reset(): AtariObsType {
    val obs = if (was_real_done)
      env.reset()
    else
      env.step(0)._1
    lives = env.unwrapped.ale.lives()
    return obs
  }
}

class FireResetEnv(env: AtariEnvType)
  : Wrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  init {
    require(env.unwrapped.action_set[1] == PLAYER_A_FIRE)
    require(env.unwrapped.action_set.size >= 3)
  }
  
  override fun reset(): NDArray<Byte> {
    env.reset()
    val (_, _, done, _) = env.step(1)
    if (done)
      env.reset()
    val (obs, _, _done, _) = env.step(2)
    if (_done)
      env.reset()
    
    return obs
  }
}

fun AtariObsType.toMat(): Mat {
  val channels = shape[2]
  return Mat(shape[0], shape[1], CV_8UC(channels),
             BytePointer(*(raw as ByteArrayBuf).raw))
}

fun Mat.toNDArray(): AtariObsType {
  require(depth() == CV_8U) { "Only supported CV_8U" }
  val channels = channels()
  val data = data()
  data.limit((rows() * cols() * channels).toLong())
//  return NDArray(Shape(rows(), cols(), channels),
//                 BytePointerBuf(data))
  return NDArray(Shape(rows(), cols(), channels),
                 ByteArray(rows() * cols() * channels) {
                   data[it.toLong()]
                 })
}

class WarpFrame(env: AtariEnvType) :
    ObservationWrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  val width = 84
  val height = 84
  
  override val observation_space =
      Box(0.toByte(), 255.toByte(), Shape(height, width, 1))
  
  override fun observation(frame: AtariObsType): AtariObsType {
    val src = frame.toMat()
    val dst = Mat()
    cvtColor(src, dst, COLOR_RGB2GRAY)
    val dst2 = Mat()
    resize(dst, dst2, Size(width, height), 0.0, 0.0, INTER_AREA)
    return dst2.toNDArray()
  }
}

class FloatFrame<WrappedEnv>(val env: AtariEnvType)
  : Env<NDArray<Float>, Float, Int, Int, WrappedEnv> {
  
  override var rand: Random = env.rand
  override val action_space = env.action_space
  override val observation_space = run {
    val obspace = env.observation_space as Box<Byte>
    Box(obspace.low.cast(NDFloat), obspace.high.cast(NDFloat))
  }
  
  override fun step(a: Int): t4<NDArray<Float>, Float, Boolean, Map<String, Any>> {
    val (_obs, reward, done, info) = env.step(a)
    
    val obs = _obs.cast(NDFloat)
    return t4(obs, reward, done, info)
  }
  
  override fun reset(): NDArray<Float> {
    val obs = env.reset()
    return obs.cast(NDFloat)
  }
  
  override fun render() = env.render()
  
  override fun close() = env.close()
  
  override fun seed(seed: Long?) = env.seed(seed)
}

class ScaledFloatFrame<WrappedEnv>(
    env: Env<NDArray<Float>, Float, Int, Int, WrappedEnv>)
  : ObservationWrapper<NDArray<Float>, Float, Int, Int, WrappedEnv>(env) {
  
  override val observation_space = Box(0f, 1f, env.observation_space.shape)
  
  override fun observation(frame: NDArray<Float>): NDArray<Float> {
    frame /= 255f
    return frame
  }
  
}

class ClipRewardEnv<WrappedEnv>(
    env: Env<NDArray<Float>, Float, Int, Int, WrappedEnv>)
  : RewardWrapper<NDArray<Float>, Float, Int, Int, WrappedEnv>(env) {
  
  override fun reward(frame: Float): Float {
    return frame.sign
  }
}

class FixedSizeDeque<E>(val capacity: Int) : ArrayDeque<E>(capacity + 1) {
  override fun addFirst(e: E) {
    if (size == capacity)
      removeLast()
    super.addFirst(e)
  }
  
  override fun addLast(e: E) {
    if (size == capacity)
      removeFirst()
    super.addLast(e)
  }
}

class FrameStack<WrappedEnv>(
    env: Env<NDArray<Float>, Float, Int, Int, WrappedEnv>, val k: Int)
  : Wrapper<NDArray<Float>, Float, Int, Int, WrappedEnv>(env) {
  
  val frames = FixedSizeDeque<NDArray<Float>>(k)
  override val observation_space = run {
    val (height, width, rgb) = env.observation_space.shape
    Box(0f, 255f, Shape(height, width, rgb * k))
  }
  val rgbDim = env.observation_space.shape[2]
  val base = observation_space.shape[2]
  
  override fun reset(): NDArray<Float> {
    val ob = env.reset()
    repeat(k) {
      frames += ob
    }
    return get_ob()
  }
  
  override fun step(a: Int): t4<NDArray<Float>, Float, Boolean, Map<String, Any>> {
    val result = env.step(a)
    frames += result._1
    result._1 = get_ob()
    return result
  }
  
  private fun get_ob(): NDArray<Float> {
    require(frames.size == k)
    val frames = frames.toList()
    val raw = FloatArray(observation_space.n) {
      val frameId = (it % base) / rgbDim
      var i = it - frameId * rgbDim
      i = i / k + i % k
      frames[frameId].rawGet(i)
    }
    return NDArray(observation_space.shape, FloatArrayBuf(raw), NDFloat)
//    return concatenate(frames, axis = 2)
  }
}

//class LazyFrames<T : Any>(val frames: List<NDArray<T>>) : NDArray<T>() {
//  lateinit var out: NDArray<T>
//  private fun force(): NDArray<T> {
//    if (!::out.isInitialized) {
//      out = concatenate(frames, axis = 2)
//    }
//    return out
//  }
//
//}