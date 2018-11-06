package org.apache.spark.example

import scala.annotation.tailrec

/**
  * Created by yihaibo on 18/10/30.
  */
object AppTest {
  def main(args: Array[String]) : Unit = {
//    val s = new BinStrategyGroupHelper(Array(0.0, 1.0, 3.0, 4.0, 5.4, 7.0))
//    println(s.getAllBinStrategy())

    val s = List(-1,2,-3,4)

    println(inc(s))
  }

  def sum(s: List[Int]): Int = {
    s match {
      case head :: tail => head + sum(tail)
      case Nil => 0
    }
  }

  @tailrec
  def inc(s: List[Int]): Boolean = {
    s match {
      case Nil => true
      case head :: Nil => true
      case head :: tail => head < tail.head && inc(tail)
    }
  }

}

