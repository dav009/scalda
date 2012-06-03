package edu.ucla.sspace.graphical

import scalanlp.stats.distributions.Gaussian

import java.io.PrintWriter


object MakeTestData {
    def main(args: Array[String]) {
        val pw = new PrintWriter(args(0))
        val means = args.view(1, args.size).map(_.split(",").map(_.toDouble).toList)
        for ( Seq(mu_x, mu_y) <- means ) {
            val xg = new Gaussian(mu_x, 1.0)
            val yg = new Gaussian(mu_y, 1.0)
            for (i <- 0 until 100)
               pw.println("%f %f".format(xg.sample, yg.sample))
        }
        pw.close
    }
}
