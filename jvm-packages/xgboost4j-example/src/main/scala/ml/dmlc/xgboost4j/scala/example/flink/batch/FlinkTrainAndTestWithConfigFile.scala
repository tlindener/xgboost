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

package ml.dmlc.xgboost4j.scala.example.flink.batch

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.example.flink.utils.{Action, ParametersUtil}
import ml.dmlc.xgboost4j.scala.flink.XGBoostModel
import ml.dmlc.xgboost4j.scala.flink.batch.XGBoost
import ml.dmlc.xgboost4j.scala.flink.utils.TrainingParameterUtils
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.ml.MLUtils

/**
  * Train and test an XGBoost model with flink batch API.
  * The configurations are read from file.
  */
object FlinkTrainAndTestWithConfigFile {
  val logger: Log = LogFactory.getLog(this.getClass)

  def main(args: Array[String]): Unit = {

    ParametersUtil.parameterCheck(args).foreach(propsPath => {
      val start = System.currentTimeMillis()
      val params = ParameterTool.fromPropertiesFile(propsPath)

      val getOptParam = ParametersUtil.getParameter(params) _
      def TrainingParameters: Map[String, Any] =
        TrainingParameterUtils.validateParameterTool(params)

      //
      // parallelism
      val numberOfParallelism = getOptParam("numberOfParallelism").toInt
      // 0 - train
      // 1 - train and save model
      // 2 - predict
      // 3 - train and predict
      val action = getOptParam("action").toInt
      require(0 to 4 contains  action,
        "The action parameter must be contained by the interval [0, 4].")
      val act = Action(action)
      logger.info(s"###PS###p;action;$act")

      val env = ExecutionEnvironment.getExecutionEnvironment

      logger.info("Default parallelism: " + env.getParallelism)
      // branching by the actions
      act match {
        // train the model
        case Action.Train =>
          val round = getOptParam("round").toInt
          val trainPath = getOptParam("trainPath")
          logger.info("result: " +
            XGBoost.train(
              MLUtils.readLibSVM(env, trainPath),
              TrainingParameters, round, numberOfParallelism)
          )
        // train and save the model
        case Action.TrainAndSave =>
          val round = getOptParam("round").toInt
          val trainPath = getOptParam("trainPath")
          val modelPath = getOptParam("modelPath")

          XGBoost.trainAndSaveModelToFile(
            MLUtils.readLibSVM(env, trainPath),
            TrainingParameters, round, numberOfParallelism, modelPath,
            saveAsHadoopFile = modelPath.startsWith(ParametersUtil.HADOOP_FILE_PREFIX))
        // predict
        case Action.Predict =>
          val modelPath = getOptParam("modelPath")
          val testPath = getOptParam("testPath")

          val prediction = XGBoost.predict(
            XGBoostModel.loadModelFromFile(modelPath,
              modelPath.startsWith(ParametersUtil.HADOOP_FILE_PREFIX)),
            MLUtils.readLibSVM(env, testPath),
            numberOfParallelism)
          logger.info("result: " + prediction.deep.mkString("[", ",", "]"))
          // test the prediction
          import ml.dmlc.xgboost4j.scala.example.util.CustomEval
          logger.info("percent of the error: " +
            new CustomEval().eval(prediction, new DMatrix(testPath).jDMatrix) * 100 + "%")
        // train the model then predict
        case Action.TrainAndPredict =>
          val round = getOptParam("round").toInt
          val trainPath = getOptParam("trainPath")
          val testPath = getOptParam("testPath")
          val prediction = XGBoost.trainAndPredict(MLUtils.readLibSVM(env, trainPath),
              MLUtils.readLibSVM(env, testPath),
            TrainingParameters, round, numberOfParallelism)

          logger.info("result: " + prediction.deep.mkString("[", ",", "]"))
          // test the prediction
          import ml.dmlc.xgboost4j.scala.example.util.CustomEval
          logger.info("percent of the error: " +
            new CustomEval().eval(prediction, new DMatrix(testPath).jDMatrix) * 100 + "%")
      }

      logger.info(s"###jobTime: ${(System.currentTimeMillis() - start) / 1000}")

    })
  }

}
