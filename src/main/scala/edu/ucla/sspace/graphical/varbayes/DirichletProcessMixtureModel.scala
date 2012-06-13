package edu.ucla.sspace.graphical.varbayes

import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Util.{logNormalize,epsilon}

import scalala.library.Library.Axis.{Horizontal,Vertical}
import scalala.library.Library.{max,sum,log,exp}
import scalala.library.Numerics.digamma
import scalala.tensor.::
import scalala.tensor.dense.{DenseMatrix,DenseVectorRow}
import scalala.tensor.mutable.VectorRow

import scala.math.{E,Pi}

import java.io.PrintWriter

class DirichletProcessMixtureModel(val nIter: Int, val alpha: Double) extends Learner {

    def train(data: List[VectorRow[Double]], k: Int,
              ignored: List[List[VectorRow[Double]]]) = {
        // Get the shape of the data set in terms of number of data points and 
        // number of features.
        val n = data.size
        val v = data(0).length

        var phi = DenseMatrix.ones[Double](n, k) / k 
        var bound_1 = -1/2d * v * log(2*Pi) - log(2*Pi* E)
        var mus = selectInitialCenters(data, k)
        var (a,b,precisions,bounds) = selectInitialPrecisions(n, v, k)
        var gamma_1 = DenseVectorRow.ones[Double](k) * alpha
        var gamma_2 = DenseVectorRow.ones[Double](k) * alpha

        for (i <- 0 until nIter) {
            printf("Iteration [%d]\n", i)
            // Compute the likelihood of each component.
            val diff = (gamma_1 + gamma_2).map(digamma)
            val gamma_1_diff = gamma_1.map(digamma) - diff
            val gamma_2_diff = DenseVectorRow.zeros[Double](k)
            gamma_2_diff(0) = digamma(gamma_2(0)) - diff(0)
            for (j <- 1 until k)
                gamma_2_diff(j) = gamma_2_diff(j-1) + digamma(gamma_2(j-1)) - diff(j-1)
            val gamma_diff = gamma_1_diff + gamma_2_diff

            // Compute the likelihood of each data point appearing in each component.
            val dataLikelihood = likelihood(data, bound_1, bounds, mus, precisions)

            // Combine the two likelihoods into the general DPMM likelihood of 
            // each data point appearing in each component.
            phi = logNormalize(dataLikelihood.mapTriples((r,c,v) => v + gamma_diff(c)))

            // Compute the bound for each data point.
            //val bound = sum(phi.mapTriples((r,c,v) => v*gamma_diff(c)), Horizontal)

            // Compute the summary of each component.
            val componentSum = sum(phi, Horizontal)

            // Update the first variation parameter to each component.
            gamma_1 = componentSum + 1

            // Update the second variation parameter to each component.  We do 
            // this backwards since the each index includes fewer sums of the 
            // components.
            gamma_2 = DenseVectorRow.fill[Double](k)(alpha)
            for (j <- k - 2 to 0 by - 1)
                gamma_2(j) = gamma_2(j+1) + componentSum(j)

            // Next update the means and precisions.
            for (j <- 0 until k) {
                val w_j = phi(::,j)
                val p_j = precisions(j)
                val center = data.zip(w_j.toList)
                                 .map{ case(x_i, w_ji) =>  x_i * w_ji }
                                 .reduce(_+_).toDense 
                mus(j) = (center * p_j) / (1d + p_j * w_j.sum)
            }

            a = sum(phi, Horizontal) * v / 2d
            for (j <- 0 until k) {
                val mu_j = mus(j)
                b(j) = 1d + data.zip(phi(::,j).toList)
                                .map{ case(x_i, w_ji) => w_ji * (((x_i - mu_j) :^2).sum + v) }
                                .sum/2d
                bounds(j) = v/2d * (digamma(a(j)) - log(b(j)))
            }
            precisions = a :/ b
        }

        mus.foreach(println)
        (0 until n).map(i => phi(i,::).argmax).toArray
    }

    def likelihood(data: List[VectorRow[Double]],
                   bound_1: Double,
                   bounds: DenseVectorRow[Double], 
                   mus: Array[DenseVectorRow[Double]],
                   precisions: DenseVectorRow[Double]) = {
        val ll = DenseMatrix.zeros[Double](data.size, mus.size)
        for (r <- 0 until ll.numRows)
            ll(r,::) += bounds + bound_1
        for (k <- 0 until ll.numCols;
             (x, r) <- data.zipWithIndex)
            ll(r,k) -= precisions(k)/2d * (((x - mus(k)) :^ 2).sum + x.length)
        ll
    }


    def selectInitialPrecisions(n: Int, v: Int, k:Int) = {
        val a = DenseVectorRow.ones[Double](k)
        val b = DenseVectorRow.ones[Double](k)
        val precisions = DenseVectorRow.ones[Double](k)
        val bounds = (a.map(digamma) :- b.map(log)) * v/2d
        (a, b, precisions, bounds)
    }

    def selectInitialCenters(data: List[VectorRow[Double]], k:Int) = {
        Array(DenseVectorRow[Double](44.7462, 41.3319),
              DenseVectorRow[Double](39.8464, 50.7417),
              DenseVectorRow[Double](20.6352, 30.2213))
        /*
        val n = data.size
        val v = data(0).length

         // Compute the global mean of all data points.
        val mu_1 = data.reduce(_+_).toDense / n
        // Compute the variance of all data points to the global mean.
        val variance_1 = (data.map(x_j => (mu_1 - x_j) :^ 2).reduce(_+_) / n)
        // Compute the standard deviation to the global mean.
        val sigma_1 = variance_1 :^ 0.5

        // Sample from a standard normal distribution to create initial means
        // for each component.  These components also have a standard variance.
        Array.fill(k)(DenseVectorRow.randn(v) :* sigma_1 :+ mu_1)
        */
    }
}
