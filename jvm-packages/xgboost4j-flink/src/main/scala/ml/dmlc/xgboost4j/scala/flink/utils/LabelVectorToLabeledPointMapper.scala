/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.flink.utils

import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.{DenseVector, SparseVector}
import org.apache.flink.streaming.api.scala.DataStream

/**
  * Helper [[MapFunction]] implementation to transform
  * [[DataStream]] of [[LabeledVector]] to [[DataStream]] of [[LabeledPoint]]
  */
class LabelVectorToLabeledPointMapper extends MapFunction[LabeledVector, LabeledPoint] {
  override def map(x: LabeledVector): LabeledPoint = {
    var index: Array[Int] = Array[Int]()
    var value: Array[Double] = Array[Double]()
    x.vector match {
      case s: SparseVector =>
        index = s.indices
        value = s.data
      case d: DenseVector =>
        val (i, v) = d.toSeq.unzip
        index = i.toArray
        value = v.toArray
    }
    LabeledPoint(x.label.toFloat,
      index, value.seq.map(z => z.toFloat).toArray)
  }
}
