package wumo.sim.tensorflow.core

import wumo.sim.util.plusAssign

class DeviceSpecification(
    val job: String = "",
    val replica: Int = -1,
    val task: Int = -1,
    val deviceType: String = "",
    val deviceIndex: Int = -1
) {
  
  /** Returns a string representation of this device, of the form:
   *
   * `/job:<name>/replica:<id>/task:<id>/device:<device_type>:<id>`.
   *
   * @return String representation of the device.
   */
  override fun toString(): String {
    val stringBuilder = StringBuilder()
    if (job.isNotEmpty())
      stringBuilder += "/job:$job"
    if (replica != -1)
      stringBuilder += "/replica:$replica"
    if (task != -1)
      stringBuilder += "/task:$task"
    if (deviceType.isNotEmpty())
      stringBuilder += if (deviceIndex != -1)
        "/device:$deviceType:$deviceIndex"
      else
        "/device:$deviceType:*"
    return stringBuilder.toString()
  }
  
  companion object {
    private val deviceStringRegex =
        """
      #^(?:/job:(?<job>[^/:]+))?
      #(?:/replica:(?<replica>[[0-9]&&[^\:]]+))?
      #(?:/task:(?<task>[[0-9]&&[^\:]]+))?
      #(?:(?:/device:(?<deviceType>[^/:]+):(?<deviceIndex>[[0-9*]&&[^\:]]+))
      #|(?:/?(?<deviceTypeShort>[^/:]+):(?<deviceIndexShort>[[0-9*]&&[^\:]]+)))?$"""
            .trimMargin("#")
            .replace("\n", "")
            .toRegex()
    
    /** Parses a [[DeviceSpecification]] specification from the provided string.
     *
     * The string being parsed must comply with the following regular expression (otherwise an
     * [[InvalidDeviceException]] exception is thrown):
     *
     * ```
     *   ^(?:/job:([^/:]+))?
     *   (?:/replica:([[0-9]&&[^\:]]+))?
     *   (?:/task:([[0-9]&&[^\:]]+))?
     *   (?:(?:/device:([^/:]+):([[0-9*]&&[^\:]]+))|(?:/([^/:]+):([[0-9*]&&[^\:]]+)))?$
     * ```
     *
     * Valid string examples:
     *   - `/job:job1/replica:1/task:22/device:CPU:0`
     *   - `/task:22/device:GPU:0`
     *   - `/CPU:1`
     *
     * @param  device String to parse.
     * @return Device specification parsed from string.
     * @throws InvalidDeviceException If the provided string does not match the appropriate regular expression.
     */
    private val wordPattern = Regex("\\d+")
    
    internal fun fromString(device: String): DeviceSpecification {
      val result = deviceStringRegex.matchEntire(device)
      if (result != null) {
        val (job, replica, task, deviceType, deviceIndex, deviceTypeShort, deviceIndexShort) = result.destructured
        if ((replica.isEmpty() || replica.matches(wordPattern))
            && (task.isEmpty() || task.matches(wordPattern))
            && (deviceIndex.isEmpty() || deviceIndex == "*" || deviceIndex.matches(wordPattern))
            && (deviceIndexShort.isEmpty() || deviceIndexShort == "*" || deviceIndexShort.matches(wordPattern)))
          return DeviceSpecification(
              job = job,
              replica = if (replica.isEmpty()) -1 else replica.toInt(),
              task = if (task.isEmpty()) -1 else task.toInt(),
              deviceType = when {
                deviceType.isEmpty() || deviceTypeShort.isEmpty() -> ""
                deviceType.isNotEmpty() -> deviceType.toUpperCase()
                else -> deviceTypeShort.toUpperCase()
              },
              deviceIndex = when {
                (deviceIndex.isEmpty() && deviceIndexShort.isEmpty())
                    || deviceIndex == "*" || deviceIndexShort == "*" -> -1
                deviceIndex.isEmpty() -> deviceIndexShort.toInt()
                else -> deviceIndex.toInt()
              }
          )
      }
      throw InvalidDeviceException("Invalid device specification '$device'.")
    }
    
    /** Merges the properties of [dev2] into those of [dev1] and returns the result as a new [DeviceSpecification].
     *
     * @param  dev1 First device specification being merged.
     * @param  dev2 Second device specification being merged.
     * @return Merged device specification.
     */
    internal fun merge(dev1: DeviceSpecification, dev2: DeviceSpecification) =
        DeviceSpecification(
            job = if (dev2.job.isNotEmpty()) dev2.job else dev1.job,
            replica = if (dev2.replica != -1) dev2.replica else dev1.replica,
            task = if (dev2.task != -1) dev2.task else dev1.task,
            deviceType = if (dev2.deviceType.isNotEmpty()) dev2.deviceType else dev1.deviceType,
            deviceIndex = if (dev2.deviceIndex != -1) dev2.deviceIndex else dev1.deviceIndex)
  }
  
}