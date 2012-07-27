package edu.ucla.sspace.graphical

import breeze.stats.distributions.Gamma
import org.apache.commons.math3.distribution.GammaDistribution
import cern.jet.random.{Gamma=>CGamma}

import scala.io.Source
import scala.math.pow

import java.io.PrintWriter


object MakeGammaData {
    def main(args: Array[String]) {
        val data = Source.fromFile(args(0)).getLines
                                           .filter(_ != "")
                                           .map(_.split("\\s+"))
                                           .map(a=>(a(0).toInt, a(1).toDouble))
                                           .toList
                                           .groupBy(_._1)
                                           .map{ case (k,v) => (k, v.map(_._2)) }
        val beta_1 = 1d /(1*1 + data(0).map(_-5.0).map(pow(_,2)).sum)
        val beta_2 = 1d /(1*1 + data(1).map(_-10.0).map(pow(_,2)).sum)
        /*
        val beta_1 = 1d + data(0).map(_-5.0).map(pow(_,2)).sum/2d
        val beta_2 = 1d + data(1).map(_-10.0).map(pow(_,2)).sum/2d
        */
        val alpha = 1d + 5000
        val pw = new PrintWriter(args(1))
        val numsamples = args(2).toInt
        for ( ((a,b), i) <- List((alpha, 1/beta_1), (alpha, 1/beta_2)).zipWithIndex) {
            val gdist = new Gamma(a, b)
            val acgDist = new GammaDistribution(a, b)

            for (x <- 0 until numsamples) {
                    pw.println("breeze %d %f".format(i, (1/gdist.sample)/10d))
                    pw.println("commons-math %d %f".format(i, (1/acgDist.sample)/10d))
                    pw.println("colt %d %f".format(i, (1/CGamma.staticNextDouble(a, 1/b))/10d))
            }
        }
        pw.close
    }
}
