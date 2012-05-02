package edu.ucla.sspace.learner.gibbs

import edu.ucla.sspace.learner.Learner

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.sparse.SparseVectorRow

import scalanlp.stats.distributions.Multinomial

import scala.math.Pi


class DirichletProcessMixtureModel(val numIterations: Int, 
                                   val alpha: Double) extends Learner {
    val epsilon = 0.000000001
    val alpha_vec = DenseVectorRow(alpha)

    def train(data: List[SparseVectorRow[Double]], ignored: Int) = {
        // Extract the shape of the data.
        val n = data.size
        val v = data(0).length
        val t = n - 1 + alpha

        // Compute the global mean of all data points.
        val mu_1 = data.reduce(_+_).toDense / n
        // Compute the variance of all data points to the global mean.
        val variance_1 = data.map(x_j => (mu_1 - x_j) :^ 2).reduce(_+_) / n
        // Compute the standard deviation to the global mean.
        val sigma_1 = variance_1 :^ 0.5

        var k = 1

        // Sample from a standard normal distribution to create initial means
        // for each component.  These components also have a standard variance.
        var mus = Array.fill(k)(DenseVectorRow.randn(v) :* sigma_1 :+ mu_1).toBuffer
        var sigmas = Array.fill(k)(DenseVectorRow.ones[Double](v)).toBuffer

        // Compute the number of assignments to each component and the summation
        // vector for each component.
        // Assign random labels to all points.
        val labels = Array.fill(n)(0).toBuffer
        var counts = DenseVectorRow.zeros[Double](k)
        counts(0) = 1

        for (i <- 0 until numIterations) {
            for ( (x_j, j) <- data.zipWithIndex ) {
                val l_j = labels(j)

                // Undo the assignment for the existing point.  This involes
                // first removing the information from the variance vectors,
                // then removing it from the center vector, then finally undoing
                // the count for the component.
                if (i != 0) {
                    sigmas(l_j) -= (mus(l_j)/counts(l_j) - x_j) :^ 2
                    mus(l_j) -= x_j
                    counts(l_j) -= 1
                }

                // Compute the probability of selecting each component based on
                // their sizes.
                val prior = DenseVectorRow.horzcat(counts, alpha_vec) / t
                // Compute the probability of the data point given each
                // component using the sufficient statistics.
                val posterior = new DenseVectorRow[Double](
                    ((counts.data, mus, sigmas).zipped.toList.map {
                        case (n_c, mu_c, sigma_c) => 
                            if (n_c < 1d) 0.0
                            else likelihood(x_j, mu_c/n_c, sigma_c/n_c) } :+
                    likelihood(x_j, mu_1, variance_1)).toArray)

                // Combine the two probabilities into a single distribution and
                // select a new label for the data point.  Record this in the
                // label array.
                val j_new = new Multinomial(norm(prior :* posterior)).sample
                labels(j) = j_new

                if (j_new == k) {
                    k += 1
                    mus.append(x_j.toDense)
                    sigmas.append(DenseVectorRow.ones[Double](v))
                    counts = DenseVectorRow.horzcat(counts, DenseVectorRow(1d))
                } else {
                    // Restore the bookeeping information for this point using the
                    // old assignment.
                    if (i != 0) {
                        counts(l_j) += 1
                        mus(l_j) += x_j
                        sigmas(l_j) += (mus(l_j)/counts(l_j) - x_j) :^ 2
                    }
                }
            }

            // Re-estimate the means, counts, and variances for each component.
            // We do this by first grouping the data points based on their
            // assigned component, summing the points assigned to each
            // component, and finally computing the variance of each point from
            // the mean.
            labels.zip(data).groupBy(_._1)
                            .map{ case(k,v) => (k, v.map(_._2)) }
                            .foreach{ case(c, x) => {
                                    mus(c) = x.reduce(_+_).toDense
                                    counts(c) = x.size
                                    sigmas(c) = x.map(x_j => (mus(c)/counts(c) - x_j) :^2).reduce(_+_) + epsilon
                                }}
        }

        // Return the labels.
        labels.toArray
    }

    /**
     * Computes the likelihood of a data point given a known mean and known
     * variance.
     */
    def likelihood(x: SparseVectorRow[Double], 
                   mu: DenseVectorRow[Double], 
                   sigma: DenseVectorRow[Double]) =  {
        1d/(pow(2*Pi, x.length/2d) * sqrt(abs(sigma.iterator.product))) *
        exp(-.5 * ((x - mu) :* (sigma :^ -1)).dot(x - mu))
    }

    /**
     * Returns the normalized version of a vector.
     */
    def norm(v: DenseVectorRow[Double]) = v / v.sum
}
