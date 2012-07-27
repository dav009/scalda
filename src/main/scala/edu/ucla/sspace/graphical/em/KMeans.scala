package edu.ucla.sspace.graphical.em

import edu.ucla.sspace.graphical.DistanceMetrics.euclidean
import edu.ucla.sspace.graphical.Learner

import breeze.linalg.DenseVector
import breeze.linalg.NormCacheVector
import breeze.linalg.Vector

import scala.util.Random


class KMedianMean(val useMean: Boolean = true) extends Learner {

    val numIterations = 100

    def train(data: List[Vector[Double]], 
              numGroups: Int,
              priorData: List[List[Vector[Double]]] = List(List[Vector[Double]]())) : Array[Int] = {
        val representative = if (useMean) findMean _ else findMedian _
        var centers: List[Vector[Double]] = Random.shuffle(data).take(numGroups).map(toDenseWithCache)
        var assignments: List[Int] = null
        for (i <- 0 until numIterations) {
            printf("Iteration [%d]\n", i)
            println("Assigning points to the nearest cluster")
            assignments = data.map( point => centers.map(euclidean(_, point)).zipWithIndex.min._2 )
            println("Computing the most similar representative")
            centers = assignments.zip(data)
                                 .groupBy(_._1)
                                 .map{ case (key, group) => representative(group.map(_._2)) }
                                 .toList
        }
        assignments.toArray
    }

    /**
    * Find the median of the data points.  This is the point with the smallest total euclidean distance to all other points in the group.
    * This current implementation runs in O(n^2) time.  There _should_ be a faster way to do it, but I'm not sure there is.
    */
    def findMedian(data: List[Vector[Double]]) : Vector[Double] = 
        toDenseWithCache(data.map(d => (data.map(euclidean(d, _)).sum, d)).minBy(_._1)._2)

    /**
    * Find the mean of the data points.  This is the cummulative vector averaged by the number of points in the group.  This runs in O(n)
    * time and can't be much faster.
    */
    def findMean(data: List[Vector[Double]]) : Vector[Double] =
        toDenseWithCache(data.reduce(_+_).map(_/data.size))

    def toDenseWithCache(vector: Vector[Double]) = {
        val d = DenseVector.zeros[Double](vector.size)
        vector.foreachPair{ case(i, v) => d(i) = v }
        new DenseVector(d.data) with NormCacheVector[Double]
    }
}
