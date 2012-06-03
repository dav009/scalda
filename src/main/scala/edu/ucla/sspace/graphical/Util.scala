package edu.ucla.sspace.graphical

import scalala.library.Library.{abs,sum,exp,log,max}
import scalala.library.Library.Axis.{Horizontal,Vertical}
import scalala.tensor.dense.DenseMatrix
import scalala.tensor.dense.DenseVectorRow


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
}
