package edu.ucla.sspace.graphical

import DistanceMetrics.euclidean
import Util.determiniate

import scalala.library.Library.{pow,sqrt,exp,log}
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scala.math.Pi


object Likelihood {
    def gaussian(x: VectorRow[Double], 
                 mu: DenseVectorRow[Double], 
                 sigma: DenseVectorRow[Double]) =  {
        val a = 1d/(pow(2*Pi, 1/2d) * sqrt(determiniate(sigma)) + Util.epsilon)
        val d = (x - mu) :^ 2
        val b = exp(-.5 * (d:/sigma).sum)
        a*b 
    }

    def gaussian(x: VectorRow[Double],
                 mu: DenseVectorRow[Double],
                 sigma_2: Double) =
        1/sqrt(sigma_2*2*Pi)*exp(-.5 * pow(euclidean(x, mu),2) / sigma_2)
}
