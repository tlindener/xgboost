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

package ml.dmlc.xgboost4j.scala.example.flink.utils

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.flink.api.java.utils.ParameterTool

/**
  * Util functions for XGBoost parameter handling.
  */
object  ParametersUtil {
  val logger: Log = LogFactory.getLog(this.getClass)
  val HADOOP_FILE_PREFIX = "hdfs:"

  /**
    * Check properties file validation.
    *
    * @param args the cli parameter
    * @return the valid properties file path
    */
  def parameterCheck(args: Array[String]): Option[String] = {
      def outputNoParamMessage(): Unit = {
        val noParamMsg = "\tUsage:\n\n\t./run <path to parameters file>"
        println(noParamMsg)
      }

      if (args.length == 0 || !new java.io.File(args(0)).exists) {
        outputNoParamMessage()
        None
      } else {
        Some(args(0))
      }
    }

  /**
    * Get and log the given mandatory parameter.
    *
    * @param params the parameters
    * @param parameterName the name of the asked parameter
    * @return the vale of the asked parameter if it is presented else throw [[RuntimeException]]
    */
    def getParameter(params: ParameterTool)(parameterName: String): String = {
      val value = params.getRequired(parameterName)
      logger.info(s"###PS###p;$parameterName;$value")
      value
    }

}
