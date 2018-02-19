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

import ml.dmlc.xgboost4j.java.{IRabitTracker, RabitTracker => PythonRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.{RabitTracker => ScalaRabitTracker}

/**
  * Rabit tracker configurations.
  *
  * @param workerConnectionTimeout The timeout for all workers to connect to the tracker.
  *                                Set timeout length to zero to disable timeout.
  *                                Use a finite, non-zero timeout value to prevent tracker from
  *                                hanging indefinitely (in milliseconds)
  *                                (supported by "scala" implementation only.)
  * @param trackerImpl Choice between "python" or "scala". The former utilizes the Java wrapper of
  *                    the Python Rabit tracker (in dmlc_core), whereas the latter is implemented
  *                    in Scala without Python components, and with full support of timeouts.
  *                    The Scala implementation is currently experimental, use at your own risk.
  */
case class TrackerConf(workerConnectionTimeout: Long, trackerImpl: String)

object TrackerUtils {

  /**
    * Get the default TrackerConf object which chooses the python implemented Tracker.
    *
    * @return the TrackerConf with the basic config
    */
  def getDefaultTackerConf: TrackerConf = TrackerConf(0L, "python")

  /**
    * Start the Tracker which is defined in the trackerConf object.
    * It will throw an IllegalArgumentException if can not be started.
    *
    * @param nWorkers is the number of parallelism of the tracked job. The number of trackers
    * @param trackerConf contains the necessary configuration to create the Tracker.
    *                    If it is not presented than the default tracker will be used.
    * @return with the IRabitTracker trait
    */
  def startTracker(nWorkers: Int,
                   trackerConf: TrackerConf = getDefaultTackerConf
                  ): IRabitTracker = {
    val tracker: IRabitTracker = trackerConf.trackerImpl match {
      case "scala" => new ScalaRabitTracker(nWorkers)
      case "python" => new PythonRabitTracker(nWorkers)
      case _ => new PythonRabitTracker(nWorkers)
    }

    require(tracker.start(trackerConf.workerConnectionTimeout), "FAULT: Failed to start tracker")
    tracker
  }

}
