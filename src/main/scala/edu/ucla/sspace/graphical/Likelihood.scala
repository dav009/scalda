package edu.ucla.sspace.graphical

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
                 sigma: Double) = 
        exp(((x-mu):^2).sum/x.length)/sqrt(2*Pi*sigma)
}
