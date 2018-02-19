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

package ml.dmlc.xgboost4j.scala.flink.batch

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait, ObjectiveTrait, XGBoost => XGBoostScala}
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.streaming.api.scala._
import org.apache.flink.util.Collector
import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.scala.flink.XGBoostModel
import ml.dmlc.xgboost4j.scala.flink.utils.{LabelVectorToLabeledPointMapper, TrackerConf, TrackerUtils, TrainingParameterUtils}
import org.apache.flink.api.scala.DataSet

object XGBoost {
  val logger: Log = LogFactory.getLog(this.getClass)

  /**
    * Helper [[RichMapPartitionFunction]] implementation to train a [[XGBoostModel]]
    * on the training data [[LabeledPoint]].
    *
    * @param paramMap the parameters of the train
    * @param round the number of training round. The round defines the tree number in the model.
    * @param workerParameters this are the initial parameters of the [[Rabit]] workers
    *                         which are responsible for the information sharing
    *                         through the threads of other model training.
    * @param obj the user-defined objective function, null by default
    * @param eval the user-defined evaluation function, null by default
    */
  private class XGBoostModelMapper(paramMap: Map[String, Any],
                                   round: Int,
                                   workerParameters: java.util.Map[String, String],
                                   obj: ObjectiveTrait = null,
                                   eval: EvalTrait = null
                           ) extends RichMapPartitionFunction[LabeledPoint, XGBoostModel] {
    val logger: Log = LogFactory.getLog(this.getClass)

    def mapPartition(it: java.lang.Iterable[LabeledPoint],
                     collector: Collector[XGBoostModel]): Unit = {
      workerParameters.put(
        "DMLC_TASK_ID", String.valueOf(this.getRuntimeContext.getIndexOfThisSubtask))
      logger.info("start with env" + workerParameters.toString)
      import scala.collection.JavaConverters.asScalaIteratorConverter
      val trainMat = new DMatrix(it.iterator.asScala)
      Rabit.init(workerParameters)
      try {
        val booster = XGBoostScala.train(
          trainMat, paramMap, round, watches = List("train" -> trainMat).toMap, obj, eval)
        collector.collect(new XGBoostModel(booster))
      } finally {
        Rabit.shutdown()
        trainMat.delete()
      }
    }
  }

  /**
    *  Train a xgboost model with [[DataSet]] of [[LabeledVector]]
    *
    * @param trainBatch the training data
    * @param params the parameters to XGBoost
    * @param round number of rounds to train
    * @param numberOfParallelism the parallelism of the training
    * @param trackerConf contains the necessary configuration to create the Tracker.
    * @param obj the user-defined objective function, null by default
    * @param eval the user-defined evaluation function, null by default
    * @return the model [[XGBoostModel]]
    */
  def train(trainBatch: DataSet[LabeledVector],
            params: Map[String, Any],
            round: Int,
            numberOfParallelism: Int,
            trackerConf: TrackerConf = TrackerUtils.getDefaultTackerConf,
            obj: ObjectiveTrait = null,
            eval: EvalTrait = null
           ): XGBoostModel = {
    require(numberOfParallelism > 0, "you must specify the number of parallelism more than 0")
    if (obj != null) {
      require(params.get("obj_type").isDefined, "parameter \"obj_type\" is not defined," +
        " you have to specify the objective type as classification or regression with a" +
        " customized objective function")
    }
    val tracker = TrackerUtils.startTracker(numberOfParallelism, trackerConf)
    val input =
      if (trainBatch.getParallelism != numberOfParallelism) {
        logger.info(s"repartitioning training set to $numberOfParallelism partitions")
        trainBatch.rebalance.map(x => x).setParallelism(numberOfParallelism)
      } else {
        trainBatch
      }

    TrainingParameterUtils.logParameters(params)

    input.map(new LabelVectorToLabeledPointMapper).setParallelism(numberOfParallelism)
      .mapPartition(new XGBoostModelMapper(
        TrainingParameterUtils.validateParameterMap(params),
        round, tracker.getWorkerEnvs, obj, eval))
      .setParallelism(numberOfParallelism)
      .collect()
      .head
  }

  /**
    * Predict with the given [[DataSet]] of [[LabeledVector]] and [[XGBoostModel]].
    *
    * @param xGBoostModel the trained model
    * @param testBatch the [[DataSet]] of [[LabeledVector]] for test
    * @param numberOfParallelism the number of parallelism of the prediction
    * @return the prediction result
    */
  def predict(xGBoostModel: XGBoostModel,
              testBatch: DataSet[LabeledVector],
              numberOfParallelism: Int): Array[Array[Float]] =
    xGBoostModel.predict(testBatch.map{ x => x.vector}, numberOfParallelism).collect().toArray

  /**
    * Train and save a xgboost model to the given file path
    * with [[DataSet]] of [[LabeledVector]].
    *
    * @param trainBatch the training data
    * @param params the parameters to XGBoost
    * @param round number of rounds to train
    * @param numberOfParallelism the parallelism of the training
    * @param filePath the path of the model to save
    * @param trackerConf contains the necessary configuration to create the Tracker
    * @param saveAsHadoopFile the file path uses hadoop filesystem or not. Default is hadoop file.
    * @param obj the user-defined objective function, null by default
    * @param eval the user-defined evaluation function, null by default
    */
  def trainAndSaveModelToFile(trainBatch: DataSet[LabeledVector],
                              params: Map[String, Any],
                              round: Int,
                              numberOfParallelism: Int,
                              filePath: String,
                              trackerConf: TrackerConf = TrackerUtils.getDefaultTackerConf,
                              saveAsHadoopFile: Boolean = true,
                              obj: ObjectiveTrait = null,
                              eval: EvalTrait = null
                             ): Unit =
    train(trainBatch, params, round, numberOfParallelism, trackerConf, obj, eval)
      .saveModel(filePath, saveAsHadoopFile)

  /**
    * Train with [[DataSet]] of [[LabeledVector]] then predict with the trained xgboost model.
    * This method offers a good and easy way to test / compare
    * the result of the training and prediction.
    *
    * @param trainBatch the training data
    * @param testBatch the test data
    * @param params the parameters to XGBoost
    * @param round number of rounds to train
    * @param numberOfParallelism the parallelism of the training
    * @param trackerConf contains the necessary configuration to create the Tracker
    * @param obj the user-defined objective function, null by default
    * @param eval the user-defined evaluation function, null by default
    * @return the prediction result
    */
  def trainAndPredict(trainBatch: DataSet[LabeledVector],
                      testBatch: DataSet[LabeledVector],
                      params: Map[String, Any],
                      round: Int,
                      numberOfParallelism: Int,
                      trackerConf: TrackerConf = TrackerUtils.getDefaultTackerConf,
                      obj: ObjectiveTrait = null,
                      eval: EvalTrait = null
                     ): Array[Array[Float]] =
    train(trainBatch, params, round, numberOfParallelism, trackerConf, obj, eval)
      .predict(testBatch.map{ x => x.vector}, numberOfParallelism).collect().toArray
}
