package edu.ucla.sspace.graphical.em

import edu.ucla.sspace.common.Statistics.bessi
import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Util.{unit,epsilon}

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scala.math.Pi
import scala.util.Random


class FiniteMixtureVonMisesFisher(numIters: Int,
                                  s: Set[Int]=Set[Int]()) extends Learner {

    type Theta = ( DenseVectorRow[Double], Double, Double)

    def train(unNormData : List[VectorRow[Double]], k: Int,
              ignored: List[List[VectorRow[Double]]]) = {
        val mean = unNormData.reduce(_+_).toDense / unNormData.size
        // Normalize each data point so that it's on the unit sphere, i.e. has magnitude 1.0
        val data = unNormData.map(_ - mean).map(x_j => x_j/x_j.norm(2))

        // Get basic dimensions of the data set.
        val n = data.size
        val v = data(0).length
        val alpha = v/2 - 1

        // Create the initial labels.
        val labels = Array.fill(n)(Random.nextInt(k))

        // Setup the initial components.  These should be semi random from the global data.  I don't know how to initialize the kappa values
        // so we'll just initialize those as 1.
        val components = computeInitialComponents(k, data, alpha, v)

        if (s contains 0)
            report(0, labels.toList)

        for (i <- 0 until numIters) {
            printf("Starting iteration [%d]\n", i)
            for ( (x_j, j) <- data.zipWithIndex) {
                def likelihood(theta: Theta) = theta._3 * exp(theta._2 * theta._1.dot(x_j))
                // Compute the expecatation for each data point based on the parameters.
                val probs = DenseVectorRow[Double](components.map(likelihood))
                labels(j) = probs.argmax
            }
            // After processing each data point, compute the maximization to estimate new parameters for each component.
            labels.zip(data)
                  .groupBy(_._1)
                  .map{ case(k,v) => (k, v.map(_._2)) }
                  .foreach{ case (c, x) => {
              val w_c = x.size / n.toDouble
              println(w_c)
              val r_c = x.reduce(_+_).toDense
              val r_c_norm = r_c.norm(2)
              val mu_c = r_c / r_c_norm
              val kappa_c = computeKappa(r_c_norm/n, v)
              val compNorm_c = w_c * computeNorm(kappa_c, alpha, v)
              components(c) = (mu_c, kappa_c, compNorm_c)
            }}
            components.foreach(theta => println(theta))

            if (s contains (i+1))
                report(i+1, labels.toList)
        }

        labels
    }

    def computeInitialComponents(k:Int, data:List[VectorRow[Double]], alpha:Int, v:Int) = {
        val r_1 = data.reduce(_+_).toDense
        val r_norm = r_1.norm(2)
        // Compute the initial mean.
        val mu_1 =  r_1 / r_norm
        // Compute the variance of all data points to the global mean.
        val variance_1 = (data.map(x_j => (mu_1 - x_j) :^ 2).reduce(_+_) / data.size) + epsilon
        // Compute the standard deviation to the global mean.
        val sigma_1 = variance_1 :^ 0.5
        val kappa_1 = computeKappa(r_norm / data.size, v)
        val compNorm = 1d/k * computeNorm(kappa_1, alpha, v)
        Array.fill(k)((unit(DenseVectorRow.randn(v) :* sigma_1 :+ mu_1), kappa_1, compNorm))
    }

    def computeKappa(r_avg: Double, v:Int) =
        (r_avg * v - pow(r_avg, 3))/(1-pow(r_avg, 2))
    def computeNorms(kappas: Array[Double], alpha: Int, v:Int) =
        DenseVectorRow[Double](kappas.map(kappa => computeNorm(kappa, alpha, v)))
    def computeNorm(kappa: Double, alpha:Int, v:Int) =
        pow(kappa, alpha)/(pow(2*Pi, v/2d)*bessi(alpha, kappa))
}
