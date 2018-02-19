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

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.SparseVector
import org.apache.flink.streaming.api.scala.{DataStream, StreamExecutionEnvironment}
import org.apache.flink.api.scala._

/** Convenience functions for machine learning tasks in streaming environment.
  *
  * This class follows the org.apache.flink.ml.MLUtils class structure and implementation
  * but this is defined on DataStream (streaming) object rather than DataSet (batch) object.
  * Due to the change from batch to stream
  * the dimension parameter should be given as a parameter too.
  *
  * This object contains convenience functions for machine learning tasks:
  *
  * - readLibSVM:
  *   Reads a libSVM/SVMLight input file and returns a data set of [[LabeledVector]].
  *   The file format is specified [http://svmlight.joachims.org/ here].
  *
  */
object MLStreamUtils {
  val logger: Log = LogFactory.getLog(this.getClass)

  /** Reads a file in libSVM/SVMLight format and converts the data into a data set of
    * [[LabeledVector]].
    * The dimension of the [[LabeledVector]] should be given as a parameter
    * due to the streaming environment.
    *
    * Since the libSVM/SVMLight format stores a vector in its sparse form, the [[LabeledVector]]
    * will also be instantiated with a [[SparseVector]].
    *
    * @param env executionEnvironment [[StreamExecutionEnvironment]]
    * @param filePath Path to the input file
    * @param dimension the dimension of the [[LabeledVector]]
    * @param numberOfParallelism parallelism of the parsing and the returning [[DataStream]]
    * @param skipBrokenLine if this boolean is true the broken format lines will be skipped.
    *                       Default value is false
    * @return [[DataStream]] of [[LabeledVector]] containing the information of the libSVM/SVMLight
    *        file
    */
  def readLibSVM(env: StreamExecutionEnvironment,
                 filePath: String,
                 dimension: Int,
                 numberOfParallelism: Int,
                 skipBrokenLine: Boolean = false
                ): DataStream[LabeledVector] = {
    env.readTextFile(filePath).setParallelism(numberOfParallelism)
      .flatMap {
        line =>
          // remove all comments which start with a '#'
          val commentFreeLine = line.takeWhile(_ != '#').trim

          if (commentFreeLine.nonEmpty) {

            try {
              val splits = commentFreeLine.split(' ')
              val label = splits.head.toDouble
              val sparseFeatures = splits.tail
              require(!sparseFeatures.isEmpty, "The line does not contains feature value.")
              val coos = sparseFeatures.map {
                str =>
                  val pair = str.split(':')
                  require(pair.length == 2,
                    "Each feature entry has to have the form <feature>:<value>")

                  // libSVM index is 1-based, but we expect it to be 0-based
                  val index = pair(0).toInt - 1
                  val value = pair(1).toDouble

                  (index, value)
              }

              Some((label, coos))

            } catch {
              case e: Exception =>
                if (skipBrokenLine) {
                  logger.debug("the following line was skipped which breaks the format: " + line)
                  None
                } else {
                  throw e
                }
            }
          } else {
            None
          }
      }.setParallelism(numberOfParallelism)
      .map(value => LabeledVector(value._1, SparseVector.fromCOO(dimension, value._2)))
      .setParallelism(numberOfParallelism)
  }

}
