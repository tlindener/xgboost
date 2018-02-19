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
import org.apache.flink.api.java.utils.ParameterTool
import scala.collection.immutable.Map
import scala.collection.immutable.HashSet
import scala.util.Try

/** Convenience functions for generate and / or validate training parameters.
  *
  */
object TrainingParameterUtils {
  val logger: Log = LogFactory.getLog(this.getClass)

  /**
    * Utility class for parsing String to T.
    * @param op parsing function
    * @tparam T the type of the output
    */
  private[flink] case class ParseOp[T](op: String => T)
  implicit val popDouble = ParseOp[Double](_.toDouble)
  implicit val popInt = ParseOp[Int](_.toInt)

  /** Parsing function: String to T. It is used for process the parameter.
    *
    * @param s the value to parse
    * @tparam T The type of the output
    * @return the parsed value of s or None if it can not be parsed to T
    */
  private def parse[T: ParseOp](s: String): Option[T] = Try {implicitly[ParseOp[T]].op(s)}.toOption

  // The supported value for some enumeration type
  val supportedBoosters = HashSet("gbtree", "gblinear", "dart")
  val supportedTreeMethods = HashSet("auto", "exact", "approx", "hist")
  val supportedGrowthPolicies = HashSet("depthwise", "lossguide")
  val supportedSampleType = HashSet("uniform", "weighted")
  val supportedNormalizeType = HashSet("tree", "forest")
  val supportedObjective = HashSet("reg:linear", "reg:logistic", "binary:logistic",
    "binary:logitraw", "count:poisson", "multi:softmax", "multi:softprob", "rank:pairwise",
    "reg:gamma")
  val supportedEvalMetrics = HashSet("rmse", "mae", "logloss", "error", "merror", "mlogloss",
    "auc", "ndcg", "map", "gamma-deviance")

  val boosterParameters = Map(
    /**
   * step size shrinkage used in update to prevents overfitting. After each boosting step, we
   * can directly get the weights of new features and eta actually shrinks the feature weights
   * to make the boosting process more conservative. [default=0.3] range: [0,1]
   */
    "eta" -> (0.3, "step size shrinkage used in update to prevents" +
        " overfitting. After each boosting step, we can directly get the weights of new features." +
        " and eta actually shrinks the feature weights " +
        "to make the boosting process more conservative.",
//      (value: Double) => value >= 0 && value <= 1),
      (value: Any) => {
        val parsed = value.asInstanceOf[Double]
        parsed >= 0 && parsed <= 1}, parse[Double] _),
    /**
   * minimum loss reduction required to make a further partition on a leaf node of the tree.
   * the larger, the more conservative the algorithm will be. [default=0] range: [0,
   * Double.MaxValue]
   */
    "gamma" -> (0d, "minimum loss reduction required to make a further" +
      " partition on a leaf node of the tree. the larger, the more conservative the algorithm" +
      " will be.",
//      (value: Double) => value >= 0),
      (value: Any) => {
        val parsed = value.asInstanceOf[Double]
        parsed >= 0}, parse[Double] _),
  /**
    * maximum depth of a tree, increase this value will make model more complex / likely to be
    * overfitting. [default=6] range: [1, Int.MaxValue]
    */
   "max_depth" -> (6, "maximum depth of a tree, increase this value" +
    " will make model more complex/likely to be overfitting.",
//     (value: Int) => value >= 1),
     (value: Any) => {
       val parsed = value.asInstanceOf[Int]
       parsed >= 1
     }, parse[Int] _),
    /**
   * Parameter of Dart booster.
   * type of normalization algorithm, options: {'tree', 'forest'}. [default="tree"]
   */
   "normalize_type" -> ("tree", "type of normalization" +
    " algorithm, options: {'tree', 'forest'}",
//    (value: String) => supportedNormalizeType.contains(value))
    (value: Any) => supportedNormalizeType.contains(value.toString), (x: String) => Some(x))
  )

  val learningTaskParameters = Map(
    /**
   * Specify the learning task and the corresponding learning objective.
   * options: reg:linear, reg:logistic, binary:logistic, binary:logitraw, count:poisson,
   * multi:softmax, multi:softprob, rank:pairwise, reg:gamma. default: reg:linear
   */
  "objective" -> ("reg:linear", "objective function used for training," +
    s" options: {${supportedObjective.mkString(",")}",
//    (value: String) => supportedObjective.contains(value))
    (value: Any) => supportedObjective.contains(value.toString), (x: String) => Some(x))

  )

//  import scala.collection.JavaConverters._
//  val defaultValues: ParameterTool = ParameterTool.fromMap(
//    (boosterDefaultParmeters.map{
//    case (k, (defaultValue, description, validateFunction)) => (k, defaultValue)} ++
//    learningTaskDefaultParams.map{
//      case (k, (defaultValue, description, validateFunction)) => (k, defaultValue)})
//      .map{case(k, v) => (k, v.toString)}.asJava
//  )

  val allParameters = boosterParameters ++ learningTaskParameters

  val defaultValues: Map[String, Any] = allParameters.map {
      case (k, (defaultValue, description, validateFunction, parseFunction)) => (k, defaultValue)
  }

  private def convertValuesFromStringToAny(parameters: Map[String, String]) =
    parameters.map{case (key, value) =>
        allParameters.get(key) match {
          case Some((defaultValue, description, validateFunction, parseFunction)) =>
            parseFunction(value) match {
              case Some(parsed) => (key -> parsed)
              case None =>
                logger.error(
                  s"The value of the ${key} parameter is in bad format or invalid value: "
                + value + ". Please use the suitable format or class.")
                throw new IllegalArgumentException
            }
          case None => (key -> value)
        }
    }

  /** Validate the given parameter map.
    * The parameters will filtered according to the existed parameters.
    * It will throw [ClassCastException] or [IllegalArgumentException]
    * if a parameter is in bad format or it has an invalid value.
    *
    * @param parameters the map of parameters
    * @return the validated parameter map or
    *         throw [ClassCastException] or [IllegalArgumentException]
    *         if a parameter is in bad format or it has an invalid value.
    */
  def validateStringParameterMap(parameters: Map[String, String]): Map[String, Any] =
    validateParameterMap(convertValuesFromStringToAny(parameters))


  /** Validate the given parameter map.
    * The parameters will filtered according to the existed parameters.
    * It will throw [ClassCastException] or [IllegalArgumentException]
    * if a parameter is in bad format or it has an invalid value.
    *
    * @param parameters the map of parameters
    * @return the validated parameter map or
    *         throw [ClassCastException] or [IllegalArgumentException]
    *         if a parameter is in bad format or it has an invalid value.
    */
  def validateParameterMap(parameters: Map[String, Any]): Map[String, Any] = {
    val validParameters = parameters.filterKeys(defaultValues.keySet)
    validParameters.foreach{case(key, value) =>
      allParameters.get(key) match {
        case Some((defaultValue, description, validateFunction, parseFunction)) =>
          try {
            require(validateFunction(value), s"The value of the ${key} parameter is not valid: " +
              value + s". Please check the valid values of the ${key} parameter " +
              s"in the documentation.")
          } catch {
            case e: ClassCastException =>
              logger.error(s"The value of the ${key} parameter is in bad format or invalid value: "
                + value + ". Please use the suitable format or class: " + e.getMessage)
          }
        case None => throw new IllegalStateException(
          "Default Parameters suffered some inner critical error.")
      }
    }
    /** Known scala issue:
      * filterKeys function on map causes the following exception
      * when we try to pass the map to a flink worker:
      * [java.io.NotSerializableException]: scala.collection.immutable.MapLike$$anon$1
      * It is documented:
      * https://issues.scala-lang.org/browse/SI-6654
      * https://issues.scala-lang.org/browse/SI-4776
      * So that explains the unnecessary ".view.force" which solves the problem.
      *
      * Note it should be updated when the basic bug would be fixed!
      */
    var ret = validParameters.view.force
    if (!ret.contains("eval_metric")) {
      ret = ret + (("eval_metric", setupDefaultEvalMetric(ret)))
    }
    if (!isClassificationTask(ret) || ret.getOrElse("numClasses", 0) == 2) {
      ret = ret - "num_class"
    }
    ret
  }

  /** Validate the given [ParameterTool].
    * The parameters will filtered according to the existed parameters.
    * It will throw [ClassCastException] or [IllegalArgumentException]
    * if a parameter is in bad format or it has an invalid value.
    *
    * @param parameters the map of parameters
    * @return the validated parameter map or
    *         throw [ClassCastException] or [IllegalArgumentException]
    *         if a parameter is in bad format or it has an invalid value.
    */
  def validateParameterTool(parameters: ParameterTool): Map[String, Any] = {
    import scala.collection.JavaConverters._
    validateStringParameterMap(parameters.toMap.asScala.toMap)
  }

  /** Get a parameter map wich contains all parameters with their default value
    *
    * @return a [Map] witch contains all parameters with their default value
    */
  def getDefaultParameters: Map[String, Any] = defaultValues

  /**
   * Explains a param.
   * @param key the name of param
   * @return a string that contains the param name, doc, and its default value
   */
  def explainParam(key: String): String =
    allParameters.get(key) match {
      case Some((defaultValue, description, validateFunction, parseFunction)) =>
        s"$key: $description ($defaultValue)"
      case None =>
        logger.warn(s"There is not any parameter with the name: " + key)
        ""
    }

  /**
    * Explains all params of this instance. See `explainParam()`.
    */
  def explainParams(): String = {
    allParameters.keySet.map(explainParam).mkString("\n")
  }

  /** Log the important settings to info or debug.
    *
    * @param parameters the parameters
    * @param debugOnly use the debug level to log or false for the info level
    */
  def logParameters(parameters: Map[String, Any], debugOnly: Boolean = false): Unit = {
    def log(x: String) = if (debugOnly) logger.debug(x)
    else logger.info(x)

    log("XGBoost training parameters:")
    parameters.filterKeys(defaultValues.keySet).foreach{
      case (key, value) => log(s"$key: $value")
    }
  }

  /** Set the default value for the parameter with the name key
    *
    * @param parameters the map of parameters
    * @param key the name of the parameter to set it to his default value
    * @return a new [Map] where the parameter with the name key is set to his default value
    *         or the parameters itself if it do not contains the given key
    */
  def setDefault(parameters: Map[String, Any], key: String): Map[String, Any] =
    defaultValues.get(key) match {
      case Some(value) => parameters + ((key, value))
      case None => parameters
    }

  /** Set the default value for all parameters which are presented in the parameter map
    *
    * @param parameters the map of parameters
    * @return a new [Map] where the parameters in the map are set to them default value
    *         or the parameters itself if it do not contains any parameter to set
    */
  def setAllDefault(parameters: Map[String, Any]): Map[String, Any] = {
    val validParameters = defaultValues.filterKeys(parameters.keySet) ++
      parameters.filterKeys(!defaultValues.keySet.contains(_))
    /** Known scala issue:
      * filterKeys function on map causes the following exception
      * when we try to pass the map to a flink worker:
      * [java.io.NotSerializableException]: scala.collection.immutable.MapLike$$anon$1
      * It is documented:
      * https://issues.scala-lang.org/browse/SI-6654
      * https://issues.scala-lang.org/browse/SI-4776
      * So that explains the unnecessary ".view.force" which solves the problem.
      *
      * Note it should be updated when the basic bug would be fixed!
      */
    validParameters.view.force
  }

  /** Determine if the training is classification or not.
    *
    * @param params the map of parameters
    * @return true if the training is classification
    */
  def isClassificationTask(params: Map[String, Any]): Boolean = {
    val objective = params.getOrElse("objective", params.getOrElse("obj_type", null))
    objective != null && {
      val objStr = objective.toString
      objStr == "classification" || objStr.startsWith("binary:") || objStr.startsWith("multi:")
    }
  }

  /** Return the suitable eval strategy based on the objective of the train.
    *
    * @param params the map of parameters
    * @return the suitable eval strategy based on the objective of the train
    */
  def setupDefaultEvalMetric(params: Map[String, Any]): String = {
    val objFunc = params.getOrElse("objective", params.getOrElse("obj_type", null))
    if (objFunc == null) {
      "rmse"
    } else {
      // compute default metric based on specified objective
      val isClassification = isClassificationTask(params)
      if (!isClassification) {
        // default metric for regression or ranking
        if (objFunc.toString.startsWith("rank")) {
          "map"
        } else {
          "rmse"
        }
      } else {
        // default metric for classification
        if (objFunc.toString.startsWith("multi")) {
          // multi
          "merror"
        } else {
          // binary
          "error"
        }
      }
    }
  }

  def main(args: Array[String]): Unit = {

    println(defaultValues)
    println(validateParameterMap(List(
      "eta" -> 1d,
      "fails" -> "test",
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap))
    println(getDefaultParameters)
    getDefaultParameters.mapValues(q => q.getClass).foreach(println)

    println("GetClass")
    validateParameterMap(List(
      "eta" -> 0.1,
      "fails" -> "test",
      "max_depth" -> 2.0,
      "objective" -> "binary:logistic").toMap)
      .mapValues(q => q.getClass).foreach(println)

    println(validateParameterMap(List(
      "eta" -> "1".toDouble,
      "fails" -> "test",
      "max_depth" -> 1,
      "objective" -> "binary:logistic").toMap))

    println(parse[Double]("1.23"))
    println(parse[Int]("1.23"))
    println(parse[Int]("1"))
    println(parse[Int]("test"))

//    println(
      convertValuesFromStringToAny(List(
      "eta" -> "0.1",
      "fails" -> "test",
      "max_depth" -> "2",
      "objective" -> "binary:logistic").toMap)
      .mapValues(q => q.getClass).foreach(println)
//    )

    import scala.collection.JavaConverters._
    validateParameterTool(ParameterTool.fromMap(List(
      "eta" -> "0.1",
      "fails" -> "test",
      "max_depth" -> "2",
      "objective" -> "binary:logistic").toMap.asJava))
    .mapValues(q => q.getClass)
      .foreach(println)

    println(explainParam("objective"))
    println(explainParam("objec"))
    println(explainParams())


    val q = List(
      "eta" -> 0.1,
      "fails" -> "test",
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap

    println(q)
    println(setDefault(q, "fails"))
    println(setDefault(q, "eta"))
    println(setAllDefault(q))
    logParameters(q)
    logParameters(q, true)
  }

}
