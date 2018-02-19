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

package ml.dmlc.xgboost4j.scala.example.flink.stream

import ml.dmlc.xgboost4j.scala.example.flink.utils.{Action, ParametersUtil}
import ml.dmlc.xgboost4j.scala.flink.stream.XGBoost
import ml.dmlc.xgboost4j.scala.flink.XGBoostModel
import ml.dmlc.xgboost4j.scala.flink.utils.{MLStreamUtils, TrainingParameterUtils}
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.streaming.api.scala._

/**
  * Train and test an XGBoost model with flink stream API.
  * The configurations are read from file.
  */
object FlinkTrainAndTestWithConfigFile {
  val logger: Log = LogFactory.getLog(this.getClass)

  def main(args: Array[String]): Unit = {

    ParametersUtil.parameterCheck(args).foreach(propsPath => {
      val params = ParameterTool.fromPropertiesFile(propsPath)

      val getOptParam = ParametersUtil.getParameter(params) _
      def TrainingParameters: Map[String, Any] =
        TrainingParameterUtils.validateParameterTool(params)

      // parallelism
      val readParallelism = getOptParam("readParallelism").toInt
      val numberOfParallelism = getOptParam("numberOfParallelism").toInt
      val skipBrokenLine = getOptParam("skipBrokenLine").toBoolean
      // 0 - train
      // 1 - train and save model
      // 2 - predict
      // 3 - train and predict
      val action = getOptParam("action").toInt
      require(0 to 4 contains  action,
        "The action parameter must be contained by the interval [0, 4].")
      val act = Action(action)
      logger.info(s"###PS###p;action;$act")
      val dimension = getOptParam("dimension").toInt

      val env = StreamExecutionEnvironment.getExecutionEnvironment

      logger.info("Default parallelism: " + env.getParallelism)
      // branching by the actions
      act match {
        // train the model
        case Action.Train =>
          XGBoost.train(
            MLStreamUtils.readLibSVM(
              env, getOptParam("trainPath"), dimension, readParallelism, skipBrokenLine),
            TrainingParameters, getOptParam("round").toInt, numberOfParallelism)
            .addSink(x => {
              logger.info("result: " + x)
            }).setParallelism(1)
        // train and save the model
        case Action.TrainAndSave =>
          XGBoost.trainAndSaveModelToFile(
            MLStreamUtils.readLibSVM(
              env, getOptParam("trainPath"), dimension, readParallelism, skipBrokenLine),
            TrainingParameters, getOptParam("round").toInt,
            numberOfParallelism, getOptParam("modelPath"),
            saveAsHadoopFile = getOptParam("modelPath").
              startsWith(ParametersUtil.HADOOP_FILE_PREFIX))
        // predict
        case Action.Predict =>
          XGBoost.predict(
            XGBoostModel.loadModelFromFile(getOptParam("modelPath"),
              getOptParam("modelPath").startsWith(ParametersUtil.HADOOP_FILE_PREFIX)),
            MLStreamUtils.readLibSVM(
              env, getOptParam("testPath"), dimension, readParallelism, skipBrokenLine),
            numberOfParallelism)
            .addSink(q => logger.info("result: " + q.deep.mkString("[", ",", "]")))
            .setParallelism(1)
        // train the model then predict
        case Action.TrainAndPredict =>
          XGBoost.trainAndPredict(
            MLStreamUtils.readLibSVM(
              env, getOptParam("trainPath"), dimension, readParallelism, skipBrokenLine),
            MLStreamUtils.readLibSVM(
              env, getOptParam("testPath"), dimension, readParallelism, skipBrokenLine),
            TrainingParameters, getOptParam("round").toInt, numberOfParallelism)
            .addSink(q => logger.info("result: " + q.deep.mkString("[", ",", "]")))
            .setParallelism(1)
      }

      println(env.getExecutionPlan)
      val start = System.currentTimeMillis()
      env.execute()
      logger.info(s"###jobTime :${(System.currentTimeMillis() - start) / 1000}")
    })
  }

}
