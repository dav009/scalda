package edu.ucla.sspace.learner

import scalanlp.stats.distributions.Gaussian

object MakeTestData {
    def main(args: Array[String]) {
        val means = List( (50.0, 50.0), (55.0, 25.0), (25.0, 25.0) )
        for ( (mu_x, mu_y) <- means ) {
            val xg = new Gaussian(mu_x, 1.0)
            val yg = new Gaussian(mu_y, 1.0)
            for (i <- 0 until 100)
               printf("%f %f\n", xg.sample, yg.sample)
        }
    }
}
