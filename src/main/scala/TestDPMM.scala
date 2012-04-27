import scalala.library.Library._
import scalala.collection.sparse.SparseArray
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.sparse.SparseVectorRow

import scala.io.Source


object TestDPMM {
    def main(args:Array[String]) {
        val data = Source.fromFile(args(0)).getLines.map( line => {
                val d = line.split("\\s+").map(_.toDouble)
                val v = new SparseVectorRow[Double](
                    new SparseArray[Double](d.size, (0 until d.size).toArray, d, d.size, d.size))
                v / v.norm(2)
        }).toList
        val dpmm = new DPMM(100, 0.5, 
                            new DenseVectorRow[Double](Array.fill(2)(0.01)),
                            0.5)
        val assignments = dpmm.train(data)
        for ( (d, l) <- data.zip(assignments))
            printf("%f %f %d\n", d(0), d(1), l)
    }
}
