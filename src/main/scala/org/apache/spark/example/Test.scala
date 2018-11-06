package org.apache.spark.example

import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.spark.sql.SparkSession

/**
  * Created by yihaibo on 18/10/29.
  */
object Test {
  def main(args: Array[String]) : Unit = {
    val appName = "spark-ext-test"
    val spark = SparkSession
      .builder
      .appName(appName).getOrCreate()

    try {
      val logData = spark.read.textFile("/etc/hosts").cache()
      val numAs = logData.filter(line => line.contains("a")).count()
      val numBs = logData.filter(line => line.contains("b")).count()
      println(s"Lines with a: $numAs, Lines with b: $numBs")
      println("success")
    }catch{
      case e : Exception =>
        val msg = ExceptionUtils.getFullStackTrace(e);
        println("failed:" + msg)
        throw e
    }
  }
}
