import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.sparse.SparseVectorRow

import scalanlp.stats.distributions.Multinomial


class DPMM(val numIterations: Int,
           val alpha: Double,
           val lambda: DenseVectorRow[Double],
           val lambda_2: Double) {

    def train(data: List[SparseVectorRow[Double]]) = {
        // Create initial parameters for the data shape and the nubmer of
        // clusters.
        val n = data.size.toDouble
        val m = data(0).length
        var k = 1

        // Create a default zero mean vector.
        val zero = new DenseVectorRow[Double](Array.fill(m)(0))

        // Create an initial count buffer to track the number of data points
        // assigned to each cluster.  This starts with nothin assigned to any
        // cluster.
        var counts = Array.fill(k)(0d).toBuffer

        // Create an initial assignment array.  At first we assume everything is
        // unassigned.
        var assignments = Array.fill(data.size)(0)

        // Create an initial mean vector which will just be the zero vector.
        var mu = Array.fill(k)(zero.toDense).toBuffer

        for (k <- 0 until numIterations) {
            for ( (x_i, i) <- data.zipWithIndex ) {
                // Undo the assignment for this current data point For the first
                // iteration, the counts are all 0 so don't alter anything in
                // that case.
                if (k != 0) {
                    val c_i_old = assignments(i)
                    counts(c_i_old) -= 1d
                    mu(c_i_old) -= x_i
                }

                // Compute the prior.  This is proportional to the number of
                // points assigned to each cluster and a small probability for a
                // new cluster.
                val prior = new DenseVectorRow[Double]((counts :+ alpha).toArray) 

                // Compute the posterior.  This is proportional to the cummulant
                // of the data point evaluated for each existing cluster plus a
                // non-existant cluster.  The cumulant is based on the summation
                // of points assigned to that cluster and the number of points
                // assigned to that cluster.
                val posterior = new DenseVectorRow[Double](
                   (mu :+ zero).zip(counts :+ 0d).map(
                       eta => cumulant(x_i, eta._1, eta._2)).toArray)

                // Create the normalized probabilities for each cluster choice
                // and sample a cluster id from the multinomial distribution.
                val probs = norm(prior :* posterior)
                println(probs.data.mkString(" "))
                val c_i_new = new Multinomial(probs).sample

                // Update the assignment for the new decision.  If the new
                // cluter id is for a brand new cluster, we just append the
                // information to counts and mu.  Otherwise, we update the
                // counts and mean vector.
                if (c_i_new == counts.size) {
                    counts.append(1d)
                    mu.append(x_i.toDense)
                } else {
                    counts(c_i_new) += 1d
                    mu(c_i_new) += x_i
                }

                // Update the assignment for this data point.
                assignments(i) = c_i_new
            }
        }
        assignments
    }

    // Computee the cumulant using a normal distribution. for 
    def cumulant(x_i: SparseVectorRow[Double],
                 mu_j: DenseVectorRow[Double],
                 n_j: Double) = {
        val a = x_i.dot(lambda + mu_j + x_i) *
                1/2d *
                (x_i * (lambda_2 + n_j + 1)).dot(x_i)
        val b = x_i.dot(lambda + mu_j) *
                1/2d *
                (x_i * (lambda_2 + n_j)).dot(x_i)
        exp(a - b)
        //printf("%f\n", x_i.dot(x_i))
        //printf("%f %f %f %f\n", a, b, z, y)
    }

    // Normalizes a vector to be a probability distribution.
    def norm(v: DenseVectorRow[Double]) = v / v.sum
}
