package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Likelihood.gaussian
import edu.ucla.sspace.graphical.Util.{norm,epsilon}

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow
import scalala.tensor.sparse.SparseVectorRow

import scalanlp.stats.distributions.Multinomial

import scala.math.Pi
import scala.util.Random


class DirichletProcessMixtureModel(val numIterations: Int, 
                                   val alpha: Double) extends Learner {
    val alpha_vec = DenseVectorRow(alpha)

    def train(data: List[VectorRow[Double]], ignored: Int) = {
        // Extract the shape of the data.
        val n = data.size
        val v = data(0).length
        val t = n - 1 + alpha

        // Compute the global mean of all data points.
        val mu_1 = data.reduce(_+_).toDense / n
        // Compute the variance of all data points to the global mean.
        val variance_1 = (data.map(x_j => (mu_1 - x_j) :^ 2).reduce(_+_) / n) + epsilon
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
        var labels = Array.fill(n)(0)
        var counts = DenseVectorRow.zeros[Double](k)
        counts(0) = 1

        for (i <- 0 until numIterations) {
            printf("Starting iteration [%d] with [%d] components,\n", i, k)
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
                val posterior = DenseVectorRow[Double](
                    ((counts.data, mus, sigmas).zipped.toList.map {
                        case (n_c, mu_c, sigma_c) => 
                            if (n_c < 1d) 0.0
                            else gaussian(x_j, mu_c/n_c, sigma_c/n_c) } :+
                    gaussian(x_j, mu_1, variance_1)).toArray)
                val probs = norm(prior :* posterior)

                // Combine the two probabilities into a single distribution and
                // select a new label for the data point.  Record this in the
                // label array.
                labels(j) = new Multinomial(probs).sample

                if (labels(j) == k) {
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
            val labelRemap = labels.zip(data)
                                   .groupBy(_._1)
                                   .map{ case(k,v) => (k, v.map(_._2)) }
                                   .zipWithIndex
                                   .map{ case ((c_old, x), c) => {
                mus(c) = x.reduce(_+_).toDense
                counts(c) = x.size
                sigmas(c) = x.map(x_j => (mus(c)/counts(c) - x_j) :^2).reduce(_+_) + epsilon
                (c_old, c)
            }}
            k = labelRemap.size
            mus = mus.slice(0, k)
            sigmas = sigmas.slice(0, k)
            counts = counts(0 until k)
            labels = labels.map(labelRemap)
        }

        // Return the labels.
        labels.toArray
    }
}
