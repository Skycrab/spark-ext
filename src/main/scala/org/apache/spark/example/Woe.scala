package org.apache.spark.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.WoeBinning
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by yihaibo on 18/10/29.
  */
object Woe {
  def main(args: Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder
      .appName("woe").getOrCreate()

    val df = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
//      .load("src/main/resources/woe.csv")
//      .load("src/main/resources/woe2.csv")
      .load("src/main/resources/woenan.csv")
      .repartition(1)
//    df.show(100)
    df.createOrReplaceTempView("woe")
    df.show(2,false)

    continuous(spark, df)
    discrete(spark, df)
  }

  def continuous(spark: SparkSession, df: DataFrame): Unit = {
    val df2= spark.sql("select * from woe where last_7_days_online_num is null")
    df2.show(2,false)

    // 连续性变量
    val woeBinning = new WoeBinning()
      .setInputCol("last_7_days_online_num")
      .setOutputCol("last_7_days_online_num_woe")
      .setLabelCol("is_bad")
      .setContinuous(true)

    val woeModel = woeBinning.fit(df)
    val result = woeModel.transform(df)
    println(woeModel.labelWoeIv.label.toList)
    println(woeModel.labelWoeIv.woeIvGroup)
    result.show()
  }

  def discrete(spark: SparkSession, df: DataFrame): Unit = {
    // 离散变量
    val woeBinning = new WoeBinning()
      .setInputCol("city_level")
      .setOutputCol("city_level_woe")
      .setLabelCol("is_bad")
      .setContinuous(false)

    val woeModel = woeBinning.fit(df)
    val result = woeModel.transform(df)
    println(woeModel.labelWoeIv.label.toList)
    println(woeModel.labelWoeIv.woeIvGroup)
    result.show()
  }
}