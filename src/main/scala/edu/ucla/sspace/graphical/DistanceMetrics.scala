package edu.ucla.sspace.graphical

import breeze.linalg._
import breeze.linalg.SparseVector.canMapPairs

import scala.math.{pow,sqrt}


object DistanceMetrics {
    def euclidean(v1: Vector[Double], v2: Vector[Double]) : Double =
        (v1, v2) match {
            case (dv1: DenseVector[Double], dv2: DenseVector[Double]) => euclidean(dv1, dv2)
            case (dv1: DenseVector[Double], sv2: SparseVector[Double]) => euclidean(dv1, sv2)
            case (sv2: SparseVector[Double], dv1: DenseVector[Double]) => euclidean(dv1, sv2)
            case (sv1: SparseVector[Double], sv2: SparseVector[Double]) => euclidean(sv1, sv2)
        }

    def euclidean(v1: DenseVector[Double], v2: DenseVector[Double]) : Double =
        sqrt(((v1-v2) :^ 2d).sum)

    def euclidean(v1: Vector[Double], v2: SparseVector[Double]) : Double = {
        var v1Magnitude = pow(v1.norm(2), 2)
        var dist = 0d
        v2.mapActivePairs( (i, y) => {
            val v1Val = v1(i)
            v1Magnitude -= v1Val * v1Val
            val diff = y - v1Val
            dist += diff * diff
            dist
        })
        sqrt(v1Magnitude + dist)
    }
}
