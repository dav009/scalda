package edu.ucla.sspace.graphical

import scala.math.sqrt
import scala.util.Random
import java.io.PrintWriter

object MakeGaussianData {
    def main(args: Array[String]) {
        val pw = new PrintWriter(args(0))
        val numsamples = args(1).toInt
        for ( ((mu, sigma_2), i) <- List((5.0, 1.0), (10.0, 6.0)).zipWithIndex) {
            val sigma = sqrt(sigma_2)
            for (x <- 0 until numsamples)
                pw.println("%d %f".format(i, Random.nextGaussian * sigma + mu))
        }
        pw.close
    }
}
