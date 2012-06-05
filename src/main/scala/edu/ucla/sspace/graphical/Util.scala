package edu.ucla.sspace.graphical

import scalala.library.Library.{abs,sum,exp,log,max}
import scalala.library.Library.Axis.{Horizontal,Vertical}
import scalala.tensor.dense.DenseMatrix
import scalala.tensor.dense.DenseVectorRow

import scala.util.Random


object Util {
    val epsilon = 1.1920929e-07

    def logNormalize(m: DenseMatrix[Double]) = {
        val bestComponent = max(m, Horizontal)
        val z = m.mapTriples( (r,c,v) => v - bestComponent(r) )
        val out = sum(z.mapValues(exp), Vertical).mapValues(log)
        val w = z.mapTriples( (r,c,v) => exp(v - out(r)) + epsilon)
        val sums = sum(w, Vertical)
        w.mapTriples( (r,c,v) => v / sums(r) ).toDense
    }

    def determiniate(sigma: DenseVectorRow[Double]) =
        abs(sigma.iterator.product) 

    def norm(v: DenseVectorRow[Double]) = v / v.sum

    def unit(v: DenseVectorRow[Double]) = v / v.norm(2)

    def sampleUnormalizedLogMultinomial(logProbs: Array[Double]) : Int = {
        val s = logProbs.foldLeft(0d)( (sum, lp) => addLog(sum, lp))
        var cut = Random.nextDouble()
        for ( (lp, i) <- logProbs.zipWithIndex ) {
            cut -= exp(lp - s)
            if (cut < 0)
                return i
        }
        return 0
    }

    def addLog(x: Double, y: Double) : Double = {
        if (x == 0)
            return y
        if (y == 0)
            return x
        if (x-y > 16)
            return x
        if (x > y) 
            return x + log(1 + exp(y-x))
        if (y-x > 16)
            return y
        return y + log(1 + exp(x-y))
    }
}
