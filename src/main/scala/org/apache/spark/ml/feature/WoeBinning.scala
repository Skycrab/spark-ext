package org.apache.spark.ml.feature

import java.{util => ju}

import org.apache.spark.SparkException
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.shared.{HasContinuous, HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.json4s._
import org.json4s.native.Serialization.write

import scala.annotation.tailrec

/**
  * WOE分箱算法
  * Created by yihaibo on 18/10/29.
  */

trait WoeBinningBase extends Params with HasInputCol with HasOutputCol with HasLabelCol with HasContinuous {

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setContinuous(value: Boolean): this.type = set(continuous, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val outputColName = $(outputCol)
    require(!schema.fieldNames.contains(outputColName), s"Output column $outputColName already exists.")
    require(schema.fieldNames.contains($(labelCol)), s"Label column $labelCol not exists")
    schema.add((StructField(outputColName, StringType, false)))
  }
}

class LabelWoeIv(val label: Array[_], val woeIvGroup: Option[WoeIvGroup]) extends Serializable {
  // 是否可以转换为woe
  val hasWoe = woeIvGroup.isDefined

  // 非null Woe
  lazy val splitWoeIvs = woeIvGroup.get.woeIvs.filter(_.position != null).toArray

  // 是否有null
  val hasNullWoe = hasWoe && !woeIvGroup.get.woeIvs.filter(_.position == null).isEmpty

  lazy val nullWoeIv = woeIvGroup.get.woeIvs.filter(_.position == null).head

  def binarySearch(splits: Array[Double], feature: Double): Int = {
    if (feature == splits.last) {
      splits.length - 2
    } else {
      val idx = ju.Arrays.binarySearch(splits, feature)
      if (idx >= 0) {
        idx
      } else {
        val insertPos = -idx - 1
        if (insertPos == 0 || insertPos == splits.length) {
          throw new SparkException(s"Feature value $feature out of Bucketizer bounds" +
            s" [${splits.head}, ${splits.last}].  Check your features, or loosen " +
            s"the lower/upper bound constraints.")
        } else {
          insertPos - 1
        }
      }
    }
  }

  // 参数类型为Double，不会处理null
  def transformContinus(feature: Any): Double = {
    var result: Double = Double.NaN
    if(hasWoe) {
      feature match {
        case null => if(hasNullWoe) result = nullWoeIv.woe
        case d: Double => {
          val idx = binarySearch(label.asInstanceOf[Array[Double]], d)
          val woeIv = splitWoeIvs.filter(_.position.contains(idx))
          if(!woeIv.isEmpty) {
            result = woeIv.head.woe
          }
        }
      }
    }
    result
  }

  def transformDiscrete(feature: String): Double = {
    val group = label.asInstanceOf[Array[String]]
    var result: Double = Double.NaN
    if(hasWoe) {
      val idx = group.indexOf(feature)
      if(idx != -1) {
        val woeIv = splitWoeIvs.filter(_.position.contains(idx))
        if(!woeIv.isEmpty) {
          result = woeIv.head.woe
        }
      }
    }
    result
  }
}

class WoeBinning(override val uid: String) extends Estimator[WoeBinningModel]
  with WoeBinningBase with DefaultParamsWritable {

  val BAD_CNT_FIELD = "badCnt"
  val GOOD_CNT_FIELD = "goodCnt"

  def this() = this(Identifiable.randomUID("woeBin"))

  @org.apache.spark.annotation.Since("2.0.0")
  override def fit(dataset: Dataset[_]): WoeBinningModel = {
    // TODO: 优化 df过小可以partition到一台机器上
    val (labels, binStrategy) = if($(continuous)) handleContinuous(dataset) else handleDiscrete(dataset)
    val woeIvGroup = binStrategy.findBestSplitStrategy()
    copyValues(new WoeBinningModel(uid, new LabelWoeIv(labels, woeIvGroup)).setParent(this))
  }

  /**
    * 类别型特, 所有取值各自为独立一箱
    * @param dataset
    */
  def handleDiscrete(dataset: Dataset[_]): (Array[_], BinStrategyDetective) = {
    val inputColName = $(inputCol)
    val labelColName = $(labelCol)

    val df = dataset.withColumn(inputColName, col(inputColName).cast(StringType)).groupBy(inputColName).
      agg(count(when(col(labelColName) === 1, true)).as(BAD_CNT_FIELD),
        count(when(col(labelColName) === 0, true)).as(GOOD_CNT_FIELD))
//    df.show(false)

    val rows = df.collect().map(s => (s.getAs[String](inputColName), s.getAs[Long](BAD_CNT_FIELD), s.getAs[Long](GOOD_CNT_FIELD)))
    val labels = rows.map(_._1).toSeq.toArray
    val splitIndex = (0 until labels.length).toArray
    val binStats = rows.map{ _ match {
      case (label, good, bad) => BinStat(labels.indexOf(label), good, bad)
    }}.toSeq.toArray

    val binStrategy = BinStrategyDetective(splitIndex, binStats)
    (labels, binStrategy)
  }

  /**
    * 数值型变量, 按照分位数分成N个箱
    * @param dataset
    */
  def handleContinuous(dataset: Dataset[_]): (Array[_], BinStrategyDetective) = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)
    val labelColName = $(labelCol)

    // null值单独处理
    val nullDf = dataset.filter(col(inputColName).isNull)
    nullDf.cache()
    val notNullDf = dataset.filter(col(inputColName).isNotNull)
    notNullDf.cache()

    // 根据分位数初始化分箱(精确模式)
    val discretizer = new QuantileDiscretizer()
      .setInputCol(inputColName)
      .setOutputCol(outputColName)
      .setNumBuckets(10)
      .setRelativeError(0)
    val bucketizer = discretizer.fit(notNullDf)
    // 初始分箱结果
    val bucketDf = bucketizer.transform(notNullDf)

    // 统计各分箱好坏数量
    val splitStatDf = bucketDf.groupBy(outputColName).agg(count(when(col(labelColName) === 1, true)).as(BAD_CNT_FIELD),
      count(when(col(labelColName) === 0, true)).as(GOOD_CNT_FIELD))
    val splitBinStats = splitStatDf.collect().map(s => BinStat(s.getAs[Double](outputColName).toInt, s.getAs[Long](BAD_CNT_FIELD), s.getAs[Long](GOOD_CNT_FIELD)))
//     splitStatDf.show(false)

    // 统计null好坏数量
    val nullBinDf = nullDf.agg(count(when(col(labelColName) === 1, true)).as(BAD_CNT_FIELD),
      count(when(col(labelColName) === 0, true)).as(GOOD_CNT_FIELD))
    val nullBinStats = nullBinDf.collect().map(s => BinStat(-1, s.getAs[Long](BAD_CNT_FIELD), s.getAs[Long](GOOD_CNT_FIELD)))
    val nullBinStat = if(nullBinStats.isEmpty) None else Option(nullBinStats(0))

    // 初始分bin结果
    val labels = bucketizer.getSplits
    val splitIndex = BinStrategyHelper.getContinuousSplitIndex(labels)

    val binStrategy = BinStrategyDetective(splitIndex, splitBinStats, nullBinStat)
    (labels, binStrategy)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): Estimator[WoeBinningModel] = {
    defaultCopy(extra)
  }
}

class WoeBinningModel(override val uid: String, val labelWoeIv: LabelWoeIv)
  extends Model[WoeBinningModel] with WoeBinningBase {

  @org.apache.spark.annotation.Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    val newCol = if($(continuous)) dataset($(inputCol)).cast(DoubleType) else dataset($(inputCol)).cast(StringType)

    val woe = if($(continuous)) {
      udf {feature: Any => labelWoeIv.transformContinus(feature)}
    } else {
      udf {feature: String => labelWoeIv.transformDiscrete(feature)}
    }
    dataset.select(col("*"), woe(newCol).as($(outputCol)))
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    if(schema.fieldNames.contains($(inputCol))) {
      validateAndTransformSchema(schema)
    }else {
      schema
    }
  }

  override def copy(extra: ParamMap): WoeBinningModel = {
    defaultCopy(extra)
  }
}

/**
  * 初始一组分bin统计结果
  * @param splitIndex  当前组序号
  * @param badCnt  当前组坏用户数量
  * @param goodCnt  当前组好用户数量
  * @param badTotal  总坏用户数量
  * @param goodTotal  总好用户数量
  */
case class BinStat(splitIndex: Int, badCnt: Long, goodCnt: Long, badTotal: Long = 0, goodTotal: Long = 0)


/**
  * 分bin一区间woe统计
  * @param position
  * @param badCnt
  * @param goodCnt
  * @param badTotal
  * @param goodTotal
  * @param woe
  * @param iv
  */
case class WoeIv(position: List[Int], badCnt: Long, goodCnt: Long,
            badTotal: Long, goodTotal: Long, woe: Double, iv: Double) extends Serializable {

  lazy val badRate = badCnt.toDouble / (badCnt+goodCnt)

  lazy val binSize = badCnt + goodCnt

  lazy val total = badTotal + goodTotal
}

/**
  * 分bin组iv统计
  */
class WoeIvGroup(val woeIvs: List[WoeIv]) extends Serializable {

  /**
    * 分bin总iv
    */
  val iv = woeIvs.map(_.iv).sum

  def add(woeIv: WoeIv): WoeIvGroup = {
    val woeIvGroup = new WoeIvGroup(woeIv +: woeIvs)
    woeIvGroup
  }

  /**
    * 当前分组binSize是否合适(排除bin中元素数量过少)
    * bin_Size_Thresh 变量分箱结果中, 每个箱中样本个数占总样本数的最小百分比阈值, 数量过少的箱会被合并
    */
  def isBinSizeSuitable(binSizeThresh: Double): Boolean = {
    val minBinSize = woeIvs.map(_.binSize).min
    minBinSize >= woeIvs.head.total * binSizeThresh
  }

  /**
    * woe是否单调(由于逻辑回归是线性模型，希望自变量与风险等级同向)
    * @return
    */
  def isMonotonous(): Boolean = {
    //单调递增
    @tailrec
    def inc(woeIvs: List[WoeIv]): Boolean = woeIvs match {
      case Nil => true
      case head :: Nil => true
      case head :: tail => head.woe < tail.head.woe && inc(tail)
    }
    //单调递减
    @tailrec
    def dec(woeIvs: List[WoeIv]): Boolean = woeIvs match {
      case Nil => true
      case head :: Nil => true
      case head :: tail => head.woe > tail.head.woe && dec(tail)
    }

    inc(woeIvs) || dec(woeIvs)
  }

  /**
    * woe是否显著(相邻bin的WOE区分度是否足够)
    * @return
    */
  def isDistinguish(woeThresh: Double): Boolean = {
    @tailrec
    def distinguish(woeIvs: List[WoeIv]) : Boolean = {
      woeIvs match {
        case Nil => true
        case head :: Nil => true
        case head :: tail => math.abs(head.woe - tail.head.woe) > woeThresh && distinguish(tail)
      }
    }
    distinguish(woeIvs)
  }

  /**
    * 获取woe不显著区间
    * @param woeThresh
    */
  def getNotDistinguishPair(woeThresh: Double): Option[(WoeIv, WoeIv)] = {
    @tailrec
    def notDistinguishPair(pair: Option[(WoeIv, WoeIv)], woeIvs: List[WoeIv]): Option[(WoeIv, WoeIv)] = {
      if(pair.isDefined) {
        return pair
      }
      woeIvs match {
        case Nil => None
        case head :: Nil => None
        case head :: tail => {
          val pair = if(math.abs(head.woe - tail.head.woe) < woeThresh) Option((head, tail.head)) else None
          notDistinguishPair(pair, tail)
        }
      }
    }
    notDistinguishPair(None, woeIvs.sortBy(_.woe))
  }

  override def toString: String = {
    implicit val formats = DefaultFormats + FieldSerializer[WoeIvGroup]()
    write(this)
  }
}

/**
  * 分bin策略
  */
trait BinStrategy {

  /**
    * 分bin个数
    */
  val binGroupSize: Int

  /**
    * 是否是null合集
    * @return
    */
  def isNull(): Boolean

  /**
    * 计算woe,iv
    * @return
    */
  def getWoeIvGroup(): WoeIvGroup

  /**
    * 获取第一个woeIv
    */
  lazy val firstWoeIv = getWoeIvGroup.woeIvs.head

  /**
    * 合并一组中两个bin
    * @return
    */
  def combine(left: WoeIv, right: WoeIv): BinStrategy
}

/**
  * 分组分bin策略
  * @param positions 本次分bin策略
  * @param splitsStat  各分bin好坏统计
  */
class SplitBinStrategy(val positions: List[List[Int]], val splitsStat: Map[Int, BinStat]) extends BinStrategy {

  override val binGroupSize: Int = positions.length

  lazy val splitKeys = splitsStat.keySet

  override def isNull(): Boolean = false

  override def getWoeIvGroup(): WoeIvGroup = {
    val goodTotal = splitsStat.head._2.goodTotal
    val badTotal = splitsStat.head._2.badTotal
    val woeIvs = positions.map{ positions =>
      val binStats = positions.filter(p => splitKeys.contains(p)).map(splitsStat.get(_).get)
      val goodCnt = binStats.map(_.goodCnt).sum
      val badCnt = binStats.map(_.badCnt).sum
      val goodCntCorrect = math.max(1, goodCnt)
      val badCntCorrect = math.max(1, badCnt)
      val gr = goodCntCorrect.toDouble / goodTotal
      val br = badCntCorrect.toDouble / badTotal
      val woe = math.log(gr/br)
      val iv = (gr-br) * woe
      WoeIv(positions, badCntCorrect, goodCntCorrect, badTotal, goodTotal, woe, iv)
    }

    new WoeIvGroup(woeIvs)
  }

  /**
    * 合并一组中两个bin
    *
    * @return
    */
  override def combine(left: WoeIv, right: WoeIv): SplitBinStrategy = {
    val p =  (left.position ::: right.position) +: positions.filter(p => p != left.position && p != right.position)
    new SplitBinStrategy(p, splitsStat)
  }
}

/**
  * null值分bin策略
  */
class NullBinStrategy(val binStat: BinStat) extends BinStrategy{

  override val binGroupSize: Int = 1

  override def isNull(): Boolean = true

  override def getWoeIvGroup(): WoeIvGroup = {
    val gr = binStat.goodCnt.toDouble / binStat.goodTotal
    val br = binStat.badCnt.toDouble / binStat.badTotal
    val woe = math.log(gr/br)
    val iv = (gr-br) * woe
    val woeIv = WoeIv(null, binStat.badCnt, binStat.goodCnt, binStat.badTotal, binStat.goodTotal, woe, iv)
    new WoeIvGroup(List(woeIv))
  }

  /**
    * 合并一组中两个bin
    *
    * @return
    */
  override def combine(left: WoeIv, right: WoeIv): NullBinStrategy = this
}

trait BinStrategyDetective {
  /**
    * 寻找最优分箱
    * @return
    */
  def findBestSplitStrategy(): Option[WoeIvGroup]
}

/**
  * 连续变量
  */
class ContinuousBinStrategyDetective(splitsIndex: Array[Int], splitsStat: Map[Int, BinStat], nullBinStat: Option[BinStat])
  extends BinStrategyDetective {
  /**
    * 全排列获取所有分组策略
    */
  lazy val splitBinStrategys: List[BinStrategy] = {
    val binStrategys: List[BinStrategy] = BinStrategyHelper.getContinuousBinStrategys(splitsIndex).
      map(s => new SplitBinStrategy(s, splitsStat))
    binStrategys
  }

  lazy val nullBinStrategy: Option[BinStrategy] = nullBinStat match {
    case Some(binStat) => Option(new NullBinStrategy(binStat))
    case None => None
  }

  /**
    * 寻找最优分箱
    *
    *  计算当前数值型特征的分箱方案, 三个条件:
    1. 是否有binSize过小(bin中元素数量过少)
    2. WOE值是否单调
    3. bin之间的WOE区分度是否足够
    */
  def findBestSplitStrategy(): Option[WoeIvGroup] = {
    val suiteWoeIvGroups = splitBinStrategys.map(_.getWoeIvGroup()).filter{ woeIvGroup =>
      woeIvGroup.isBinSizeSuitable(WoeConfig.BinSizeThresh) &&
        woeIvGroup.isMonotonous() &&
        woeIvGroup.isDistinguish(WoeConfig.WoeThresh)
    }

    var bestWoeIvGroup: WoeIvGroup = null
    if(!suiteWoeIvGroups.isEmpty) {
      // 取IV最大值
      bestWoeIvGroup = suiteWoeIvGroups.maxBy(_.iv)
      if(nullBinStrategy.isDefined) {
        bestWoeIvGroup = bestWoeIvGroup.add(getBestNullWoeIv(bestWoeIvGroup))
      }
    }
    Option(bestWoeIvGroup)
  }

  def getBestNullWoeIv(splitWoeIvGroup: WoeIvGroup): WoeIv = {
    val nullWoeIvGroup = nullBinStrategy.get.getWoeIvGroup()
    val nullWoeIv = nullWoeIvGroup.woeIvs.head
    // 如果空值的bin_size过小, 则用woe最相近的非空区间代替空值的WOE
    if(!nullWoeIvGroup.isBinSizeSuitable(WoeConfig.BinSizeThresh)) {
      val minWoe = splitWoeIvGroup.woeIvs.minBy(w => math.abs(w.woe-nullWoeIv.woe)).woe
      nullWoeIv.copy(woe = minWoe)
    }else {
      nullWoeIv
    }
  }
}

/**
  * 离散变量
  */
class DiscreteBinStrategyDetective(splitsIndex: Array[Int], splitsStat: Map[Int, BinStat]) extends BinStrategyDetective {
  /**
    * 初始每个离散值即为单独分箱
    */
  lazy val initBinStrategy: BinStrategy = {
    val positions = for(index <- splitsIndex.toList) yield List(index)
    new SplitBinStrategy(positions, splitsStat)
  }

  /**
    * 寻找最优分箱
    *
    * 计算离散变量的的分箱方案, 两个条件:
    1. 所有箱的size(样本个数)不能小于阈值bin_size_thresh
    2. 各箱的woe不能过于相似: 差别不能小于阈值woe_thresh
    * @return
    */
  override def findBestSplitStrategy(): Option[WoeIvGroup] = {
    var binStrategy = initBinStrategy
    var finished = false
    while (binStrategy.binGroupSize >=2 && !finished) {
      val woeIvGroup = binStrategy.getWoeIvGroup()
      // binSize过小, 合并最小的两个bin
      if(!woeIvGroup.isBinSizeSuitable(WoeConfig.BinSizeThresh)) {
        val woeIvs = woeIvGroup.woeIvs.sortBy(_.binSize)
        binStrategy = binStrategy.combine(woeIvs.head, woeIvs.tail.head)
      }else {
        // 合并不显著woe区间(woe过于接近)
        val pair = woeIvGroup.getNotDistinguishPair(WoeConfig.WoeThresh)
        if(pair.isDefined) {
          val (left, right) = pair.get
          binStrategy = binStrategy.combine(left, right)
        }else {
          finished = true
        }
      }
    }
    Option(binStrategy.getWoeIvGroup())
  }
}

object BinStrategyDetective {
  def buildStat(splitBinStats: Array[BinStat], nullBinStat: Option[BinStat]) = {
    val (nullBadCnt, nullGoodCnt) = nullBinStat match {
      case Some(binStat) => (binStat.badCnt, binStat.goodCnt)
      case None => (0L, 0L)
    }

    val badTotal = splitBinStats.map(_.badCnt).sum
    val goodTotal = splitBinStats.map(_.goodCnt).sum
    val splitsStat: Map[Int, BinStat] = splitBinStats.map(b =>
      (b.splitIndex, b.copy(badTotal=badTotal + nullBadCnt, goodTotal=goodTotal + nullGoodCnt))
    ).toMap

    val nullStat = nullBinStat match {
      case Some(binStat) => {
        // 防止计算WOE的时候分子或者分母为0, 造成WOE无穷大
        val nullBadCntCorrect = math.max(1, nullBadCnt)
        val nullGoodCntCorrect = math.max(1, nullGoodCnt)
        Option(BinStat(-1, nullBadCntCorrect, nullGoodCntCorrect, badTotal + nullBadCntCorrect, goodTotal + nullGoodCntCorrect))
      }
      case None => None
    }
    (splitsStat, nullStat)
  }

  /**
    * 连续变量
    * @param splitsIndex  初始分bin结果索引，如分为3个bin -> [0, 1, 2]
    * @param splitBinStats  初始分bin统计好坏统计信息(可能有些bin无统计信息)
    * @param nullBinStat  空值bin单独统计信息
    * @return
    */
  def apply(splitsIndex: Array[Int], splitBinStats: Array[BinStat], nullBinStat: Option[BinStat]): BinStrategyDetective = {
    val (splitsStat, nullStat) = buildStat(splitBinStats, nullBinStat)
    val binStrategyGroup = new ContinuousBinStrategyDetective(splitsIndex, splitsStat, nullStat)
    binStrategyGroup
  }

  /**
    * 离散变量
    */
  def apply(splitsIndex: Array[Int], splitBinStats: Array[BinStat]): BinStrategyDetective = {
    val (splitsStat, _) = buildStat(splitBinStats, None)
    val binStrategyGroup = new DiscreteBinStrategyDetective(splitsIndex, splitsStat)
    binStrategyGroup
  }
}

object WoeConfig {
  /** 分箱算法中, 类别型特征的最大独特值个数, 取值数太多的类别型特征会被跳过而不进行分 **/
  val CategoricalMaxEnum : Int = 50

  /** 分箱算法中, 数值型变量的初始分箱个数 **/
  val InitBinCntNumeric : Int = 10

  /** 变量分箱结果中, 每个箱中样本个数占总样本数的最小百分比阈值, 数量过少的箱会被合并 **/
  val BinSizeThresh : Double = 0.01

  /** 变量分箱结果中, 每个箱的WOE值与相邻箱的WOE值最小差值阈值, WOE值区别过小的箱会被合并 **/
  val WoeThresh = 0.03

}

/**
  * 分bin策略辅助类
  */
object BinStrategyHelper {

  /**
    * splits(连续变量初始分bin结果) -> List(-Infinity, 0.0, 1.0, 3.0, 4.0, 6.0, 7.0, Infinity)
    *
    * splitsIndex ->  位置[0, 1, 2, 3, 4, 5, 6]
    *
   */
  def getContinuousSplitIndex(splits: Array[Double]): Array[Int] = {
    val splitsIndex = 0 to splits.length - 2
    splitsIndex.toArray
  }

  /**
    * 连续变量所有分bin结果
    *
    * 遍历所有可能分bin方案,共2的n-1次方中
    *
    * splits(连续变量初始分bin结果) -> List(-Infinity, 0.0, 1.0, 3.0, 4.0, 6.0, 7.0, Infinity)
    *
    * splitsIndex ->  位置[0, 1, 2, 3, 4, 5, 6]
    *
    * 只能放6个隔板
    *
    * 假如 splitsIndex : [0, 1, 2, 3, 4]
      [
        [[0], [1, 2, 3, 4]],
        [[0, 1], [2, 3, 4]],
        [[0, 1, 2], [3, 4]],
        [[0, 1, 2, 3], [4]],
        [[0], [1], [2, 3, 4]],
        [[0], [1, 2], [3, 4]],
        [[0], [1, 2, 3], [4]],
        [[0, 1], [2], [3, 4]],
        [[0, 1], [2, 3], [4]],
        [[0, 1, 2], [3], [4]],
        [[0], [1], [2], [3, 4]],
        [[0], [1], [2, 3], [4]],
        [[0], [1, 2], [3], [4]],
        [[0, 1], [2], [3], [4]],
        [[0], [1], [2], [3], [4]]
      ]
    */
  def getContinuousBinStrategys(splitsIndex: Array[Int]): List[List[List[Int]]] = {
    val spilitsLength = splitsIndex.length

    // 隔板可能出现位置(1, N-1)
    val boardPositions = 1 until spilitsLength
    // 可能的隔板数量
    val boardNums = 1 until spilitsLength

    var strategys = List[List[List[Int]]]()
    for(num <- boardNums) {
      for(positions <- boardPositions.combinations(num)) {
        // 可能的隔板位置
        var current = List[List[Int]]()
        // 切片开始位置
        var start = 0
        // 把当前隔板方案中, 在尾插入最后隔板(方便迭代转化)
        val indexs =  positions :+ spilitsLength
        for(index <- indexs) {
          current = current :+ splitsIndex.slice(start, index).toList
          start = index
        }
        strategys = current +: strategys
      }
    }
    strategys
  }
}