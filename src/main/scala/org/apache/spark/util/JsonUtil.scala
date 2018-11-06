package org.apache.spark.util

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization._


/**
  * Created by yihaibo on 18/10/30.
  */
object JsonUtil {
  def encode(any : AnyRef): String = {
    implicit val formats = Serialization.formats(NoTypeHints)
    write(any)
  }

}
