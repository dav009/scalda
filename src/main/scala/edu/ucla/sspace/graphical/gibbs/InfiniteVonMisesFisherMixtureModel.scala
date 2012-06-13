package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.common.Statistics.bessi
import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Util.{norm,epsilon}

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Multinomial

import scala.math.Pi
import scala.util.Random


class InfiniteVonMisesFisherMixtureModel(numIters: Int,
                                         alpha: Double,
                                         s: Set[Int]=Set[Int]()) extends Learner {

    type Theta = (Double, DenseVectorRow[Double], Double, Double)

    def train(unNormData : List[VectorRow[Double]], ignored: Int,
              ingored2: List[List[VectorRow[Double]]]) = {
        // Normalize each data point so that it's on the unit sphere, i.e. has magnitude 1.0
        val mean = unNormData.reduce(_+_).toDense / unNormData.size
        val data = unNormData.map(_ - mean).map(x_j => x_j/x_j.norm(2))

        // Get basic dimensions of the data set.
        val n = data.size
        val nd = n.toDouble
        val t = n - 1 + alpha
        val v = data(0).length
        val order = v/2 - 1

        // Setup the initial components.  These should be semi random from the global data.  I don't know how to initialize the kappa values
        // so we'll just initialize those as 1.
        val components = Array(computeParameters(data, alpha, nd, order, v)).toBuffer
        val kappa_base = components(0)._3

        // Create the initial labels.
        var labels = Array.fill(n)(0)

        if (s contains 0)
            report(0, labels.toList)

        for (i <- 0 until numIters) {
            printf("Starting iteration [%d] with [%d] components,\n", i, components.size-1)

            for ( (x_j, j) <- data.zipWithIndex) {
                val l_j = labels(j)
                if (i != 0)
                    components(l_j) = updateComponent(components(l_j), -1)

                def likelihood(theta: Theta) = theta._4 * exp(theta._3 * theta._2.dot(x_j))

                // Compute the expecatation for each data point based on the parameters.
                val prior = DenseVectorRow[Double](components.map(_._1 / t).toArray)
                val posterior = DenseVectorRow[Double](components.map(likelihood).toArray)
                val l_j_new  = new Multinomial(norm(prior:*posterior)).sample

                if (l_j_new == 0) {
                    // If the global component was created, create a new 
                    // component using just the current data point.
                    labels(j) = components.size
                    components.append(newComponent(x_j, kappa_base, order, v))
                } else {
                    // Restore the bookeeping information for this point using the
                    // old assignment.
                    labels(j) = l_j_new
                    components(l_j_new) = updateComponent(components(l_j_new), +1)
                }
            }
            // After processing each data point, compute the maximization to estimate new parameters for each component.
            val labelRemap = labels.zip(data)
                                   .groupBy(_._1)
                                   .map{ case(k,v) => (k, v.map(_._2)) }
                                   .zipWithIndex
                                   .map{ case ((c_old, x), c) => {
                components(c+1) = computeParameters(x.toList, x.size, nd, order, v)
                (c_old, c+1)
            }}
            components.trimEnd(components.size - (labelRemap.size+1))
            labels = labels.map(labelRemap)

            if (s contains (i+1))
                report(i+1, labels.toList)
        }

        labels
    }

    def newComponent(x: VectorRow[Double], kappa: Double, order:Int, v:Int) = {
        val k = Random.nextDouble * kappa
        (1d, x.toDense, k, computeNorm(k, order, v))
    }

    def computeParameters(data:List[VectorRow[Double]], smoothing: Double, n: Double, order:Int, v:Int) = {
        val r_1 = data.reduce(_+_).toDense
        val r_norm = r_1.norm(2)
        val mu_1 =  r_1 / r_norm
        val kappa_1 = computeKappa(r_norm / n , v)
        val compNorm_1 = computeNorm(kappa_1, order, v)
        (smoothing, mu_1, kappa_1, compNorm_1)
    }

    def computeKappa(r_avg: Double, v:Int) =
        (r_avg * v - pow(r_avg, 3))/(1-pow(r_avg, 2))
    def computeNorms(kappas: Array[Double], order: Int, v:Int) =
        DenseVectorRow[Double](kappas.map(kappa => computeNorm(kappa, order, v)))
    def computeNorm(kappa: Double, order:Int, v:Int) =
        pow(kappa, order)/(pow(2*Pi, v/2d)*bessi(order, kappa))
    def updateComponent(theta: Theta, delta:Int) =
        (theta._1+delta, theta._2, theta._3, theta._4)
}
