// All credit goes to https://github.com/yusugomori/DeepLearning
// $ scalac LogisticRegression.scala
// $ scala LogisticRegression

import scala.math

class LogisticRegression(val N: Int, val n_in: Int, val n_out: Int) {

  val W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)
  val b: Array[Double] = new Array[Double](n_out)

  def train(x: Array[Double], y: Array[Int], lr: Double) {
    val p_y_given_x: Array[Double] = new Array[Double](n_out)
    val dy: Array[Double] = new Array[Double](n_out)

    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_out) {
      p_y_given_x(i) = 0
      for(j <- 0 until n_in) {
        p_y_given_x(i) += W(i)(j) * x(j)
      }
      p_y_given_x(i) += b(i)
    }
    softmax(p_y_given_x)

    for(i <- 0 until n_out) {
      dy(i) = y(i) - p_y_given_x(i)

      for(j <- 0 until n_in) {
        W(i)(j) += lr * dy(i) * x(j) / N
      }
      b(i) += lr * dy(i) / N
    }
  }


  def softmax(x: Array[Double]) {
    var max: Double = 0.0
    var sum: Double = 0.0

    var i: Int = 0
    for(i <- 0 until n_out) if(max < x(i)) max = x(i)

    for(i <- 0 until n_out) {
      x(i) = math.exp(x(i) - max)
      sum += x(i)
    }

    for(i <- 0 until n_out) x(i) /= sum
  }


  def predict(x: Array[Double], y: Array[Double]) {
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_out) {
      y(i) = 0
      for(j <- 0 until n_in) {
        y(i) += W(i)(j) * x(j)
      }
      y(i) += b(i)
    }
    softmax(y)
  }

}


object LogisticRegression {
  def test_lr() {
    
    val learning_rate: Double = 1
    val n_epochs: Int = 20000

    val fea = scala.io.Source.fromFile("C:\\...\\iono_fea.csv")
    val train_X: Array[Array[Double]] = fea.getLines.map(line => {val tokens = line.split(","); 
    tokens.map(_.toDouble)}).toArray
    fea.close()

   // val train_X: Array[Array[Double]] = Array(
   //   Array(1, 1, 1, 0, 0, 0),
   //   Array(1, 0, 1, 0, 0, 0),
   //   Array(1, 1, 1, 0, 0, 0),
   //   Array(0, 0, 1, 1, 1, 0),
   //   Array(0, 0, 1, 0, 1, 0),
   //   Array(0, 0, 1, 1, 1, 0)
   // )

    val tar = scala.io.Source.fromFile("C:\\...\\iono_tar.csv")
    val train_Y: Array[Array[Int]] = tar.getLines.map(line => {val tokens = line.split(","); 
    tokens.map(_.toInt)}).toArray
    tar.close()

    //val train_Y: Array[Array[Int]] = Array(
    //  Array(1,0),
    //  Array(1,0),
    //  Array(1,0),
    //  Array(0,1),
    //  Array(0,1),
    //  Array(0,1)
    //)

    val train_N: Int = train_X.length
    val n_in: Int = 34
    val n_out: Int = 2

    // construct
    val classifier = new LogisticRegression(train_N, n_in, n_out)

    // train
    var epoch: Int = 0
    var i: Int = 0
    for(epoch <- 0 until n_epochs) {
      for(i <- 0 until train_N) {
        classifier.train(train_X(i), train_Y(i), learning_rate)
      }
      // learning_rate *= 0.95
    }
    
    val test_X: Array[Array[Double]] = train_X
    // test data
    //val test_X: Array[Array[Double]] = Array(
    //  Array(1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.0376,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.3409,0.42267,-0.54487,0.18641,-0.453),
    //  Array(1,0,1,-0.18829,0.93035,-0.36156,-0.10868,-0.93597,1,-0.04549,0.50874,-0.67743,0.34432,-0.69707,-0.51685,-0.97515,0.05499,-0.62237,0.33109,-1,-0.13151,-0.453,-0.18056,-0.35734,-0.20332,-0.26569,-0.20468,-0.18401,-0.1904,-0.11593,-0.16626,-0.06288,-0.13738,-0.02447)
    //)

    val test_N: Int = test_X.length

    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_out)

    // test
    
    var j: Int = 0
    for(i <- 0 until test_N) {
      classifier.predict(test_X(i), test_Y(i))
      for(j <- 0 until n_out) {
        printf("%.5f ", test_Y(i)(j))
      }
      println()
    }

     val pred: Array[Double] = Array.ofDim[Double](test_N)
     val trained: Array[Double] = Array.ofDim[Double](test_N) 
     for(i <- 0 until test_N) {
      if (test_Y(i)(0) >= test_Y(i)(1) )
         pred(i) = 1
      else pred(i) = 0 

      if (train_Y(i)(0) >= train_Y(i)(1) )
         trained(i) = 1.0
      else trained(i) = 0.0 
     }
     
     val diff_train = Array.ofDim[(Double,Double)](test_N)
     for (i<-0 until test_N) { diff_train(i) = (pred(i),trained(i))}
     val NtrainErr = diff_train.filter(r=>r._1!=r._2).length.toDouble
     val trainErr = NtrainErr / test_N
     println("Number of Misclassification = " + NtrainErr + " " + "Training Error = " + trainErr)
      //println(pred.mkString(" "))
      //println(train_Y.mkString(" "))
      //println(trained.mkString(" "))
      //println(diff_train.mkString(" "))
  }

  def main(args: Array[String]) {
    test_lr()
  }

}
