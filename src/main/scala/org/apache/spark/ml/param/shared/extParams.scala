package org.apache.spark.ml.param.shared

import org.apache.spark.ml.param.{BooleanParam, Params}

/**
  * Created by yihaibo on 18/11/2.
  */
class extParams {

}


/**
  * Trait for shared param continuous variable (default: true).
  */
trait HasContinuous extends Params {
  /**
    * Param for whether the independent variable is continuous variable.
    */
  final val continuous : BooleanParam = new BooleanParam(this, "continuous", "whether the independent variable is continuous variable")

  setDefault(continuous, true)

  final def getContinuous: Boolean = $(continuous)
}

