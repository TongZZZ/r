// All credit goes to yusugomori 
import scala.util.Random
import scala.math

class HiddenLayer(val N: Int, val n_in: Int, val n_out: Int, _W: Array[Array[Double]], _b: Array[Double], var rng: Random=null) {


  def uniform(min: Double, max: Double): Double = {
    return rng.nextDouble() * (max - min) + min
  }

  def binomial(n: Int, p: Double): Int = {
    if(p < 0 || p > 1) return 0

    var c: Int = 0
    var r: Double = 0.0

    var i: Int = 0

    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }

    return c
  }

  def sigmoid(x: Double): Double = {
    return 1.0 / (1.0 + math.pow(math.E, -x))
  }


  if(rng == null) rng = new Random(1234)

  var a: Double = 0.0
  var W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)
  var b: Array[Double] = new Array[Double](n_out)

  var i: Int = 0
  if(_W == null) {
    a = 1.0 / n_in

    for(i <- 0 until n_out) {
      for(j <- 0 until n_in) {
        W(i)(j) = uniform(-a, a)
      }
    }
  } else {
    W = _W
  }

  if(_b != null) b = _b


  def output(input: Array[Int], w: Array[Double], b: Double): Double = {
    var linear_output: Double = 0.0

    var j: Int = 0
    for(j <- 0 until n_in) {
      linear_output += w(j) * input(j)
    }
    linear_output += b

    return sigmoid(linear_output)
  }

  def sample_h_given_v(input: Array[Int], sample: Array[Int]) {
    var i: Int = 0
    
    for(i <- 0 until n_out) {
      sample(i) = binomial(1, output(input, W(i), b(i)))
    }
  }
}


class LogisticRegression(val N: Int, val n_in: Int, val n_out: Int) {

  val W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)
  val b: Array[Double] = new Array[Double](n_out)

  def train(x: Array[Int], y: Array[Int], lr: Double) {
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


  def predict(x: Array[Int], y: Array[Double]) {
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

class dA(val N: Int, val n_visible: Int, val n_hidden: Int,
         _W: Array[Array[Double]]=null, _hbias: Array[Double]=null, _vbias: Array[Double]=null,
         var rng: Random=null) {
  
  var W: Array[Array[Double]] = Array.ofDim[Double](n_hidden, n_visible)
  var hbias: Array[Double] = new Array[Double](n_hidden)
  var vbias: Array[Double] = new Array[Double](n_visible)


  if(rng == null) rng = new Random(1234)

  if(_W == null) {
    var i: Int = 0
    var j: Int = 0

    val a: Double = 1 / n_visible
    for(i <- 0 until n_hidden)
      for(j <- 0 until n_visible)
        W(i)(j) = uniform(-a, a)

  } else {
    W = _W
  }

  if(_hbias == null) {
    var i: Int = 0
    for(i <- 0 until n_hidden) hbias(i) = 0
  } else {
    hbias = _hbias
  }

  if(_vbias == null) {
    var i: Int = 0
    for(i <- 0 until n_visible) vbias(i) = 0
  } else {
    vbias = _vbias
  }


  def uniform(min: Double, max: Double): Double = rng.nextDouble() * (max - min) + min
  def binomial(n: Int, p: Double): Int = {
    if(p < 0 || p > 1) return 0
    
    var c: Int = 0
    var r: Double = 0

    var i: Int = 0
    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }

    c
  }

  def sigmoid(x: Double): Double = 1.0 / (1.0 + math.pow(math.E, -x))


  def get_corrupted_input(x: Array[Int], tilde_x: Array[Int], p: Double) {
    var i: Int = 0;
    for(i <- 0 until n_visible) {
      if(x(i) == 0) {
        tilde_x(i) = 0;
      } else {
        tilde_x(i) = binomial(1, p)
      }
    }
  }

  // Encode
  def get_hidden_values(x: Array[Int], y: Array[Double]) {
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_hidden) {
      y(i) = 0
      for(j <- 0 until n_visible) {
        y(i) += W(i)(j) * x(j)
      }
      y(i) += hbias(i)
      y(i) = sigmoid(y(i))
    }
  }

  // Decode
  def get_reconstructed_input(y: Array[Double], z: Array[Double]) {
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_visible) {
      z(i) = 0
      for(j <- 0 until n_hidden) {
        z(i) += W(j)(i) * y(j)
      }
      z(i) += vbias(i)
      z(i) = sigmoid(z(i))
    }
  }

  def train(x: Array[Int], lr: Double, corruption_level: Double) {
    var i: Int = 0
    var j: Int = 0

    val tilde_x: Array[Int] = new Array[Int](n_visible)
    val y: Array[Double] = new Array[Double](n_hidden)
    val z: Array[Double] = new Array[Double](n_visible)

    val L_vbias: Array[Double] = new Array[Double](n_visible)
    val L_hbias: Array[Double] = new Array[Double](n_hidden)

    val p: Double = 1 - corruption_level

    get_corrupted_input(x, tilde_x, p)
    get_hidden_values(tilde_x, y)
    get_reconstructed_input(y, z)

    // vbias
    for(i <- 0 until n_visible) {
      L_vbias(i) = x(i) - z(i)
      vbias(i) += lr * L_vbias(i) / N
    }

    // hbias
    for(i <- 0 until n_hidden) {
      L_hbias(i) = 0
      for(j <- 0 until n_visible) {
        L_hbias(i) += W(i)(j) * L_vbias(j)
      }
      L_hbias(i) *= y(i) * (1 - y(i))
      hbias(i) += lr * L_hbias(i) / N
    }

    // W
    for(i <- 0 until n_hidden) {
      for(j <- 0 until n_visible) {
        W(i)(j) += lr * (L_hbias(i) * tilde_x(j) + L_vbias(j) * y(i)) / N
      }
    }
  }

  def reconstruct(x: Array[Int], z: Array[Double]) {
    val y: Array[Double] = new Array[Double](n_hidden)

    get_hidden_values(x, y)
    get_reconstructed_input(y, z)
  }

}



class SdA(val N: Int, val n_ins: Int, hidden_layer_sizes: Array[Int], val n_outs: Int, val n_layers:Int, var rng: Random=null) {

  def sigmoid(x: Double): Double = {
    return 1.0 / (1.0 + math.pow(math.E, -x))
  }

  var input_size: Int = 0

  // var hidden_layer_sizes: Array[Int] = new Array[Int](n_layers)
  var sigmoid_layers: Array[HiddenLayer] = new Array[HiddenLayer](n_layers)
  var dA_layers: Array[dA] = new Array[dA](n_layers)

  if(rng == null) rng = new Random(1234)


  var i: Int = 0

  // construct multi-layer
  for(i <- 0 until n_layers) {
    if(i == 0) {
      input_size = n_ins
    } else {
      input_size = hidden_layer_sizes(i-1)
    }
    
    // construct sigmoid_layer
    sigmoid_layers(i) = new HiddenLayer(N, input_size, hidden_layer_sizes(i), null, null, rng)

    // construct dA_layer
    dA_layers(i) = new dA(N, input_size, hidden_layer_sizes(i), sigmoid_layers(i).W, sigmoid_layers(i).b, null, rng)
  }

  // layer for output using LogisticRegression
  val log_layer = new LogisticRegression(N, hidden_layer_sizes(n_layers-1), n_outs)


  def pretrain(train_X: Array[Array[Int]], lr: Double, corruption_level: Double, epochs: Int) {
    var layer_input: Array[Int] = new Array[Int](0)
    var prev_layer_input_size: Int = 0
    var prev_layer_input: Array[Int] = new Array[Int](0)

    var i: Int = 0
    var j: Int = 0
    var epoch: Int = 0
    var n: Int = 0
    var l: Int = 0

    for(i <- 0 until n_layers) {  // layer-wise
      for(epoch <- 0 until epochs) {  // training epochs
        for(n <- 0 until N) {  // input x1...xN
          // layer input
          for(l <- 0 to i) {
            if(l == 0) {
              layer_input = new Array[Int](n_ins)
              for(j <- 0 until n_ins) layer_input(j) = train_X(n)(j)
            } else {
              if(l == 1) prev_layer_input_size = n_ins
              else prev_layer_input_size = hidden_layer_sizes(l-2)

              prev_layer_input = new Array[Int](prev_layer_input_size)
              for(j <- 0 until prev_layer_input_size) prev_layer_input(j) = layer_input(j)

              layer_input = new Array[Int](hidden_layer_sizes(l-1))
              
              sigmoid_layers(l-1).sample_h_given_v(prev_layer_input, layer_input)
            }
          }

          dA_layers(i).train(layer_input, lr, corruption_level)
        }
      }
    }
    
  }


  def finetune(train_X: Array[Array[Int]], train_Y: Array[Array[Int]], lr: Double, epochs: Int) {
    var layer_input: Array[Int] = new Array[Int](0)
    var prev_layer_input: Array[Int] = new Array[Int](0)

    var epoch: Int = 0
    var n: Int = 0
    
    
    for(epoch <- 0 until epochs) {
      for(n <- 0 until N) {
        
        // layer input
        for(i <- 0 until n_layers) {
          if(i == 0) {
            prev_layer_input = new Array[Int](n_ins)
            for(j <- 0 until n_ins) prev_layer_input(j) = train_X(n)(j)
          } else {
            prev_layer_input = new Array[Int](hidden_layer_sizes(i-1))
            for(j <- 0 until hidden_layer_sizes(i-1)) prev_layer_input(j) = layer_input(j)
          }

          layer_input = new Array[Int](hidden_layer_sizes(i))
          sigmoid_layers(i).sample_h_given_v(prev_layer_input, layer_input)
        }

        log_layer.train(layer_input, train_Y(n), lr)
      }
      // lr *= 0.95
    }
  }

  def predict(x: Array[Int], y: Array[Double]) {
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input: Array[Double] = new Array[Double](n_ins)
    
    var j: Int = 0
    for(j <- 0 until n_ins) prev_layer_input(j) = x(j)

    var linear_output: Double = 0.0

    // layer activation
    var i: Int = 0
    var k: Int = 0

    for(i <- 0 until n_layers) {
      layer_input = new Array[Double](sigmoid_layers(i).n_out)

      for(k <- 0 until sigmoid_layers(i).n_out) {
        linear_output = 0.0

        for(j <- 0 until sigmoid_layers(i).n_in) {
          linear_output += sigmoid_layers(i).W(k)(j) * prev_layer_input(j)
        }
        linear_output += sigmoid_layers(i).b(k)
        layer_input(k) = sigmoid(linear_output)
      }

      if(i < n_layers-1) {
        prev_layer_input = new Array[Double](sigmoid_layers(i).n_out)
        for(j <- 0 until sigmoid_layers(i).n_out) prev_layer_input(j) = layer_input(j)
      }
    }

    for(i <- 0 until log_layer.n_out) {
      y(i) = 0
      for(j <- 0 until log_layer.n_in) {
        y(i) += log_layer.W(i)(j) * layer_input(j)
      }
      y(i) += log_layer.b(i)
    }

    log_layer.softmax(y)
  }

}


object SdA {
  def test_sda() {
    val rng: Random = new Random(123)
    
    val pretrain_lr: Double = 0.1
    val corruption_level: Double = 0.3
    val pretraining_epochs: Int = 1000
    val finetune_lr: Double = 0.1
    val finetune_epochs: Int = 500

    val train_N: Int = 10
    val test_N: Int = 4
    val n_ins: Int = 28
    val n_outs: Int = 2
    val hidden_layer_sizes: Array[Int] = Array(15, 15)
    val n_layers: Int = hidden_layer_sizes.length

    // training data
    val train_X: Array[Array[Int]] = Array(
			Array(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1)
    )

    val train_Y: Array[Array[Int]] = Array(
			Array(1, 0),
			Array(1, 0),
			Array(1, 0),
			Array(1, 0),
			Array(1, 0),
			Array(0, 1),
			Array(0, 1),
			Array(0, 1),
			Array(0, 1),
			Array(0, 1)
    )
    
    // construct SdA
    val sda:SdA = new SdA(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, rng)

    // pretrain
    sda.pretrain(train_X, pretrain_lr, corruption_level, pretraining_epochs)

    // finetune
    sda.finetune(train_X, train_Y, finetune_lr, finetune_epochs)
      
    // test data
    val test_X: Array[Array[Int]] = Array(
			Array(1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
    )

    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_outs)

    // test
    var i: Int = 0
    var j: Int = 0

    for(i <- 0 until test_N) {
      sda.predict(test_X(i), test_Y(i))
      for(j <- 0 until n_outs) {
        print(test_Y(i)(j) + " ")
      }
      println()
    }
  }

  def main(args: Array[String]) {
    test_sda()
  }
}
