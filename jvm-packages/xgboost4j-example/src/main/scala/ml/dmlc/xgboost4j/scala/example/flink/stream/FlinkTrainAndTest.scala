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
import ml.dmlc.xgboost4j.scala.flink.XGBoostModel
import ml.dmlc.xgboost4j.scala.flink.stream.XGBoost
import ml.dmlc.xgboost4j.scala.flink.utils.MLStreamUtils
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.flink.streaming.api.scala._

/**
  * Train and test an XGBoost model with flink stream API.
  */
object FlinkTrainAndTest {
  val logger: Log = LogFactory.getLog(this.getClass)

  def main(args: Array[String]) {
    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
    // parallelism of the file read
    val readParallelism = 1
    // number of parallelism
    val numberOfParallelism = 1
    // dimension of the training data
    val dimension = 126
    // read training data
    val trainPath = "/home/lukacsg/git/xgboost/demo/data/agaricus.txt.train"
//        "/path/to/data/agaricus.txt.train"
    val testPath = "/home/lukacsg/git/xgboost/demo/data/agaricus.txt.test"
//      "/path/to/data/agaricus.txt.test"
    val modelPath = "/data/lukacsg/sandbox/xgboosttestmodel/model"
    // define parameters
    val paramMap = List(
      "eta" -> 0.1,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap
    // number of iterations
    val round = 2
    // choose your desired action
    // 0 - train
    // 1 - train and save model
    // 2 - predict
    // 3 - train and predict
    val action = 3
    require(0 to 4 contains  action,
      "The action parameter must be contained by the interval [0, 4].")
    val act = Action(action)
    // branching by the actions
    act match {
      // train the model
      case Action.Train =>
        XGBoost.train(
          MLStreamUtils.readLibSVM(env, trainPath, dimension, readParallelism),
          paramMap, round, numberOfParallelism)
          .addSink(x => {
            logger.info("result: " + x)
          }).setParallelism(1)
      // train and save the model
      case Action.TrainAndSave =>
        XGBoost.trainAndSaveModelToFile(
          MLStreamUtils.readLibSVM(env, trainPath, dimension, readParallelism),
          paramMap, round, numberOfParallelism, modelPath,
          saveAsHadoopFile = modelPath.startsWith(ParametersUtil.HADOOP_FILE_PREFIX))
      // predict
      case Action.Predict =>
        XGBoost.predict(
          XGBoostModel.loadModelFromFile(modelPath,
            modelPath.startsWith(ParametersUtil.HADOOP_FILE_PREFIX)),
          MLStreamUtils.readLibSVM(env, testPath, dimension, readParallelism),
          numberOfParallelism)
          .addSink(q => logger.info("result: " + q.deep.mkString("[", ",", "]"))).setParallelism(1)
      // train the model then predict
      case Action.TrainAndPredict =>
        XGBoost.trainAndPredict(
          MLStreamUtils.readLibSVM(env, trainPath, dimension, readParallelism),
          MLStreamUtils.readLibSVM(env, testPath, dimension, readParallelism),
          paramMap, round, numberOfParallelism)
          .addSink(q => logger.info("result: " + q.deep.mkString("[", ",", "]")))
          .setParallelism(1)
    }

    println(env.getExecutionPlan)
    val start = System.currentTimeMillis()
    env.execute()
    logger.info(s"###jobTime :${(System.currentTimeMillis() - start) / 1000}")
  }
}
