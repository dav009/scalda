import scalanlp.stats.distributions.Gaussian

object MakeTestData {
    def main(args: Array[String]) {
        val means = List( (1.0, 2.0), (1.0, -2.0), (-1.0, 0.0) )
        for ( (mu_x, mu_y) <- means ) {
            val xg = new Gaussian(mu_x, 1.0)
            val yg = new Gaussian(mu_y, 1.0)
            for (i <- 0 until 100)
               printf("%f %f\n", xg.sample, yg.sample)
        }
    }
}
