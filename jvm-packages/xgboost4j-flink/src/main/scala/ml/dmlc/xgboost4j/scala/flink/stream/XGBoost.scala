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

package ml.dmlc.xgboost4j.scala.flink.stream

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.scala.flink.XGBoostModel
import ml.dmlc.xgboost4j.scala.flink.utils.{LabelVectorToLabeledPointMapper, TrackerConf, TrackerUtils, TrainingParameterUtils}
import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait, ObjectiveTrait, XGBoost => XGBoostScala}
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.flink.api.common.functions.RichFlatMapFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.{DenseVector, SparseVector, Vector}
import org.apache.flink.streaming.api.scala.{DataStream, _}
import org.apache.flink.util.Collector

import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer

object XGBoost {
  val logger: Log = LogFactory.getLog(this.getClass)

  /**
    * Helper [[RichFlatMapFunction]] implementation to train a [[XGBoostModel]]
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
                                  ) extends RichFlatMapFunction[LabeledPoint, XGBoostModel] {
    val logger: Log = LogFactory.getLog(this.getClass)

    var collector: Option[Collector[XGBoostModel]] = _
    val points: ArrayBuffer[LabeledPoint] = ArrayBuffer.empty[LabeledPoint]

    override def flatMap(value: LabeledPoint, out: Collector[XGBoostModel]): Unit = {
      collector = Some(out)
      points += value
    }

    override def close(): Unit = {
      super.close()
      collector match {
        case Some(c) =>
          workerParameters.put("DMLC_TASK_ID",
            String.valueOf(this.getRuntimeContext.getIndexOfThisSubtask))
          logger.info("start with env" + workerParameters.toString)
          val trainMat = new DMatrix(points.iterator)
          Rabit.init(workerParameters)
          try {
            val booster = XGBoostScala.train(trainMat, paramMap, round,
              watches = List("train" -> trainMat).toMap, obj, eval)
            c.collect(new XGBoostModel(booster))
          } finally {
            Rabit.shutdown()
            trainMat.delete()
          }
        case None => logger.error("There is not any collector")
      }
    }
  }

  /**
    * Train a xgboost model with [[DataStream]] of [[LabeledVector]].
    *
    * @param trainStream the training data
    * @param params the parameters to XGBoost
    * @param round number of rounds to train
    * @param numberOfParallelism the parallelism of the training
    * @param trackerConf contains the necessary configuration to create the Tracker.
    * @param obj the user-defined objective function, null by default
    * @param eval the user-defined evaluation function, null by default
    * @return the model [[XGBoostModel]] in a [[DataStream]]
    */
  def train(trainStream: DataStream[LabeledVector],
            params: Map[String, Any],
            round: Int,
            numberOfParallelism: Int,
            trackerConf: TrackerConf = TrackerUtils.getDefaultTackerConf,
            obj: ObjectiveTrait = null,
            eval: EvalTrait = null
           ): DataStream[XGBoostModel] = {
    require(numberOfParallelism > 0, "you must specify the number of parallelism more than 0")
    if (obj != null) {
      require(params.get("obj_type").isDefined, "parameter \"obj_type\" is not defined," +
        " you have to specify the objective type as classification or regression with a" +
        " customized objective function")
    }
    val tracker = TrackerUtils.startTracker(numberOfParallelism, trackerConf)
    val input =
      if (trainStream.parallelism != numberOfParallelism) {
        logger.info(s"repartitioning training set to $numberOfParallelism partitions")
        trainStream.rebalance.map(x => x).setParallelism(numberOfParallelism)
      } else {
        trainStream
      }

    TrainingParameterUtils.logParameters(params)

    input.forward.map(new LabelVectorToLabeledPointMapper)
      .setParallelism(numberOfParallelism)
      .flatMap(new XGBoostModelMapper(
        TrainingParameterUtils.validateParameterMap(params),
        round, tracker.getWorkerEnvs, obj, eval))
      .setParallelism(numberOfParallelism)
      .flatMap(new RichFlatMapFunction[XGBoostModel, XGBoostModel] {
        var collector: Option[Collector[XGBoostModel]] = _
        val models: ArrayBuffer[XGBoostModel] = ArrayBuffer.empty[XGBoostModel]

        override def flatMap(value: XGBoostModel, out: Collector[XGBoostModel]): Unit = {
          collector = Some(out)
          models += value
        }

        override def close(): Unit = {
          super.close()
          collector match {
            case Some(c) =>
              import scala.collection.JavaConverters._
              Rabit.init(
                Array("DMLC_TASK_ID" -> getRuntimeContext.getIndexOfThisSubtask.toString
                ).toMap.asJava)
              c.collect(models.head)
              Rabit.shutdown()
            case None => logger.error("There is not any collector")
          }
        }
      }).setParallelism(1)
  }

  /**
    * Predict with the given [[DataStream]] of [[LabeledVector]] and [[XGBoostModel]].
    *
    * @param xGBoostModel the trained model
    * @param testStream the [[DataStream]] of [[LabeledVector]] for test
    * @param numberOfParallelism the number of parallelism of the prediction
    * @return the prediction result [[DataStream]] of Array[Array[Float]\]
    */
  def predict(xGBoostModel: XGBoostModel,
              testStream: DataStream[LabeledVector],
              numberOfParallelism: Int
             ): DataStream[Array[Array[Float]]] = {
    // LabeledVector form test data are transformed to LabeledPoint
    val testData = testStream
      .rebalance.map(x => x).setParallelism(numberOfParallelism)
      .forward.map(x => LabeledVector(0d, x.vector)).setParallelism(numberOfParallelism)
      .map(new LabelVectorToLabeledPointMapper).setParallelism(numberOfParallelism)
      .flatMap(new RichFlatMapFunction[LabeledPoint, DMatrix] {
        val logger: Log = LogFactory.getLog(this.getClass)

        var collector: Option[Collector[DMatrix]] = None
        val points: ArrayBuffer[LabeledPoint] = ArrayBuffer.empty[LabeledPoint]

        override def flatMap(value: LabeledPoint, out: Collector[DMatrix]): Unit = {
          collector = Some(out)
          points += value
        }

        override def close(): Unit = {
          super.close()
          collector match {
            case Some(c) =>
              c.collect(new DMatrix(points.iterator))
            case None => logger.error("There is not any collector")
          }
        }
      }).setParallelism(numberOfParallelism)

    xGBoostModel.predict(testData, numberOfParallelism)
  }

  /**
    * Train and save a xgboost model to the given file path
    * with [[DataStream]] of [[LabeledVector]].
    *
    * @param trainStream the training data
    * @param params the parameters to XGBoost
    * @param round number of rounds to train
    * @param numberOfParallelism the parallelism of the training
    * @param filePath the path of the model to save
    * @param trackerConf contains the necessary configuration to create the Tracker
    * @param saveAsHadoopFile the file path uses hadoop filesystem or not. Default is hadoop file.
    * @param obj the user-defined objective function, null by default
    * @param eval the user-defined evaluation function, null by default
    */
  def trainAndSaveModelToFile(trainStream: DataStream[LabeledVector],
                              params: Map[String, Any],
                              round: Int,
                              numberOfParallelism: Int,
                              filePath: String,
                              trackerConf: TrackerConf = TrackerUtils.getDefaultTackerConf,
                              saveAsHadoopFile: Boolean = true,
                              obj: ObjectiveTrait = null,
                              eval: EvalTrait = null
                             ): Unit =
    train(trainStream, params, round, numberOfParallelism, trackerConf, obj, eval)
      .addSink(_.saveModel(filePath, saveAsHadoopFile)).setParallelism(1)

  /**
    * Train with [[DataStream]] of [[LabeledVector]] then predict with the trained xgboost model.
    * This method offers a good and easy way to test / compare
    * the result of the training and prediction.
    *
    * @param trainStream the training data
    * @param testStream the test data
    * @param params the parameters to XGBoost
    * @param round number of rounds to train
    * @param numberOfParallelism the parallelism of the training
    * @param trackerConf contains the necessary configuration to create the Tracker
    * @param obj the user-defined objective function, null by default
    * @param eval the user-defined evaluation function, null by default
    * @return the prediction result [[DataStream]] of Array[Array[Float]\]
    */
  def trainAndPredict(trainStream: DataStream[LabeledVector],
                      testStream: DataStream[LabeledVector],
                      params: Map[String, Any],
                      round: Int,
                      numberOfParallelism: Int,
                      trackerConf: TrackerConf = TrackerUtils.getDefaultTackerConf,
                      obj: ObjectiveTrait = null,
                      eval: EvalTrait = null
                     ): DataStream[Array[Array[Float]]] = {
    // train the model
    val modelDataSource =
      train(trainStream, params, round, numberOfParallelism, trackerConf, obj, eval)

    // LabeledVector form test data are transformed to LabeledPoint
    val testData = testStream
      .rebalance.map(x => x).setParallelism(numberOfParallelism)
      .forward.map(x => LabeledVector(0d, x.vector)).setParallelism(numberOfParallelism)
      .map(new LabelVectorToLabeledPointMapper).setParallelism(numberOfParallelism)

    // predict
    type UnionT = Either[LabeledPoint, XGBoostModel]
    val test: DataStream[UnionT] = testData.forward.map(Left(_))
    val model: DataStream[UnionT] = modelDataSource.forward.map(Right(_))

    test.setParallelism(numberOfParallelism).union(model.setParallelism(1))
      .rebalance.map(x => x).setParallelism(1)
      .flatMap(new RichFlatMapFunction[UnionT, Array[Array[Float]]] {
        val logger: Log = LogFactory.getLog(this.getClass)

        var model: Option[XGBoostModel] = None
        var collector: Option[Collector[Array[Array[Float]]]] = _
        val points: ArrayBuffer[LabeledPoint] = ArrayBuffer.empty[LabeledPoint]

        override def flatMap(value: UnionT, out: Collector[Array[Array[Float]]]): Unit = {
          collector = Some(out)
          value match {
            case Left(point) =>
              points += point
            case Right(newModel) =>
              model = model match {
                case None => Some(newModel)
                case Some(oldModel) =>
                  logger.error("There must be exist only one model in this point. " +
                    "Old model: " + oldModel)
                  Some(newModel)
              }
          }
        }

        override def close(): Unit = {
          super.close()
          collector match {
            case Some(c) =>
              model match {
                case Some(m) =>
                  if (points.nonEmpty) {
                    c.collect(m.getBooster.predict(new DMatrix(points.iterator)))
                  }
                case None => logger.error("There is not any model")
              }
            case None => logger.error("There is not any collector")
          }
        }
      }).setParallelism(1)
  }

  /**
    * Predict with the given [[DataStream]] of ([[DMatrix]], Array[K]) and [[XGBoostModel]].
    * The [[DataStream]] contains the test data and a K type key for the data too
    * in order to make the identification of the prediction result easier.
    *
    * @param xGBoostModel the trained model
    * @param testStream The [[DataStream]] of ([[DMatrix]], Array[K]) for test
    * @param numberOfParallelism the number of parallelism of the prediction
    * @tparam K the type of the key
    * @return the prediction result [[DataStream]] of Array[(Array[Float], K)] with the key
    */
  def predictWithId[K: ClassTag](xGBoostModel: XGBoostModel,
                                 testStream: DataStream[(LabeledVector, K)],
                                 numberOfParallelism: Int
                               ): DataStream[Array[(Array[Float], K)]] = {
    // type information for serialization of flink
    implicit val typeInfo = TypeInformation.of(classOf[(LabeledVector, K)])
    implicit val typeInfo2 = TypeInformation.of(classOf[(Vector, K)])
    implicit val typeInfo3 = TypeInformation.of(classOf[(LabeledPoint, K)])
    implicit val typeInfo4 = TypeInformation.of(classOf[(DMatrix, Array[K])])
    // LabeledVector form test data are transformed to LabeledPoint
    val testData = testStream
      .rebalance.map(x => x).setParallelism(numberOfParallelism)
      .forward.map(x => (x._1.vector, x._2)).setParallelism(numberOfParallelism).map(x => {
      var index: Array[Int] = Array[Int]()
      var value: Array[Double] = Array[Double]()
      x._1 match {
        case s: SparseVector =>
          index = s.indices
          value = s.data
        case d: DenseVector =>
          val (i, v) = d.toSeq.unzip
          index = i.toArray
          value = v.toArray
      }
      (LabeledPoint(0.0f,
        index, value.seq.map(z => z.toFloat).toArray), x._2)
    }).setParallelism(numberOfParallelism)
      .flatMap(new RichFlatMapFunction[(LabeledPoint, K), (DMatrix, Array[K])] {
        val logger: Log = LogFactory.getLog(this.getClass)

        var collector: Option[Collector[(DMatrix, Array[K])]] = None
        val points: ArrayBuffer[LabeledPoint] = ArrayBuffer.empty[LabeledPoint]
        val ids: ArrayBuffer[K] = ArrayBuffer.empty[K]

        override def flatMap(value: (LabeledPoint, K),
                             out: Collector[(DMatrix, Array[K])]): Unit = {
          collector = Some(out)
          points += value._1
          ids += value._2
        }

        override def close(): Unit = {
          super.close()
          collector match {
            case Some(c) =>
              c.collect((new DMatrix(points.iterator), ids.toArray))
            case None => logger.error("There is not any collector")
          }
        }
      }).setParallelism(numberOfParallelism)

    xGBoostModel.predictWithId(testData, numberOfParallelism)
  }
}
