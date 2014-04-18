// All credit goes to https://github.com/yusugomori/DeepLearning

import scala.util.Random
import scala.math

class LogisticRegression(val N: Int, val n_in: Int, val n_out: Int) {

  val W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)
  val b: Array[Double] = new Array[Double](n_out)

  def train(x: Array[Double], y: Array[Double], lr: Double) {
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


  def output(input: Array[Double], w: Array[Double], b: Double): Double = {
    var linear_output: Double = 0.0

    var j: Int = 0
    for(j <- 0 until n_in) {
      linear_output += w(j) * input(j)
    }
    linear_output += b

    return sigmoid(linear_output)
  }

  def sample_h_given_v(input: Array[Double], sample: Array[Double]) {
    var i: Int = 0
    
    for(i <- 0 until n_out) {
      sample(i) = binomial(1, output(input, W(i), b(i)))
    }
  }
}


class RBM(val N: Int, val n_visible: Int, val n_hidden: Int,
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

  
  def contrastive_divergence(input: Array[Double], lr: Double, k: Int) {
    val ph_mean: Array[Double] = new Array[Double](n_hidden)
    val ph_sample: Array[Double] = new Array[Double](n_hidden)
    val nv_means: Array[Double] = new Array[Double](n_visible)
    val nv_samples: Array[Double] = new Array[Double](n_visible)
    val nh_means: Array[Double] = new Array[Double](n_hidden)
    val nh_samples: Array[Double] = new Array[Double](n_hidden)

    /* CD-k */
    sample_h_given_v(input, ph_mean, ph_sample)

    var step: Int = 0
    for(step <- 0 until k) {
      if(step == 0) {
        gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples)
      } else {
        gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples)
      }
    }

    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_hidden) {
      for(j <- 0 until n_visible) {
        // W(i)(j) += lr * (ph_sample(i) * input(j) - nh_means(i) * nv_samples(j)) / N
        W(i)(j) += lr * (ph_mean(i) * input(j) - nh_means(i) * nv_samples(j)) / N
      }
      hbias(i) += lr * (ph_sample(i) - nh_means(i)) / N
    }

    for(i <- 0 until n_visible) {
      vbias(i) += lr * (input(i) - nv_samples(i)) / N
    }
  }


  def sample_h_given_v(v0_sample: Array[Double], mean: Array[Double], sample: Array[Double]) {
    var i: Int = 0
    for(i <- 0 until n_hidden) {
      mean(i) = propup(v0_sample, W(i), hbias(i))
      sample(i) = binomial(1, mean(i))
    }
  }

  def sample_v_given_h(h0_sample: Array[Double], mean: Array[Double], sample: Array[Double]) {
    var i: Int = 0
    for(i <- 0 until n_visible) {
      mean(i) = propdown(h0_sample, i, vbias(i))
      sample(i) = binomial(1, mean(i))
    }
  }

  def propup(v: Array[Double], w: Array[Double], b: Double): Double = {
    var pre_sigmoid_activation: Double = 0
    var j: Int = 0
    for(j <- 0 until n_visible) {
      pre_sigmoid_activation += w(j) * v(j)
    }
    pre_sigmoid_activation += b
    sigmoid(pre_sigmoid_activation)
  }

  def propdown(h: Array[Double], i: Int, b: Double): Double = {
    var pre_sigmoid_activation: Double = 0
    var j: Int = 0
    for(j <- 0 until n_hidden) {
      pre_sigmoid_activation += W(j)(i) * h(j)
    }
    pre_sigmoid_activation += b
    sigmoid(pre_sigmoid_activation)
  }

  def gibbs_hvh(h0_sample: Array[Double], nv_means: Array[Double], nv_samples: Array[Double], nh_means: Array[Double], nh_samples: Array[Double]) {
    sample_v_given_h(h0_sample, nv_means, nv_samples)
    sample_h_given_v(nv_samples, nh_means, nh_samples)
  }


  def reconstruct(v: Array[Double], reconstructed_v: Array[Double]) {
    val h: Array[Double] = new Array[Double](n_hidden)
    var pre_sigmoid_activation: Double = 0
    
    var i: Int = 0
    var j: Int = 0
    
    for(i <- 0 until n_hidden) {
      h(i) = propup(v, W(i), hbias(i))
    }

    for(i <- 0 until n_visible) {
      pre_sigmoid_activation = 0
      for(j <- 0 until n_hidden) {
        pre_sigmoid_activation += W(j)(i) * h(j)
      }
      pre_sigmoid_activation += vbias(i)
      reconstructed_v(i) = sigmoid(pre_sigmoid_activation)
    }
  }
}


class DBN(val N: Int, val n_ins: Int, hidden_layer_sizes: Array[Int], val n_outs: Int, val n_layers: Int, var rng: Random=null) {

  def sigmoid(x: Double): Double = {
    return 1.0 / (1.0 + math.pow(math.E, -x))
  }


  var input_size: Int = 0
  
  val sigmoid_layers: Array[HiddenLayer] = new Array[HiddenLayer](n_layers)
  val rbm_layers: Array[RBM] = new Array[RBM](n_layers)

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

    // construct rbm_layer
    rbm_layers(i) = new RBM(N, input_size, hidden_layer_sizes(i), sigmoid_layers(i).W, sigmoid_layers(i).b, null, rng)

  }

  // layer for output using LogisticRegression
  val log_layer: LogisticRegression = new LogisticRegression(N, hidden_layer_sizes(n_layers-1), n_outs)


  def pretrain(train_X: Array[Array[Double]], lr: Double, k: Int, epochs: Int) {
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input_size: Int = 0
    var prev_layer_input: Array[Double] = new Array[Double](0)
    
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
              layer_input = new Array[Double](n_ins)
              for(j <- 0 until n_ins) layer_input(j) = train_X(n)(j)

            } else {
              if(l == 1) prev_layer_input_size = n_ins
              else prev_layer_input_size = hidden_layer_sizes(l-2)

              prev_layer_input = new Array[Double](prev_layer_input_size)
              for(j <- 0 until prev_layer_input_size) prev_layer_input(j) = layer_input(j)

              layer_input = new Array[Double](hidden_layer_sizes(l-1))
              sigmoid_layers(l-1).sample_h_given_v(prev_layer_input, layer_input)
            }
          }

          rbm_layers(i).contrastive_divergence(layer_input, lr, k)
        }
      }
    }
  }


  def finetune(train_X: Array[Array[Double]], train_Y: Array[Array[Double]], lr: Double, epochs: Int) {
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input: Array[Double] = new Array[Double](0)

    var epoch: Int = 0
    var n: Int = 0
    var i: Int = 0
    var j: Int = 0

    for(epoch <- 0 until epochs) {
      for(n <- 0 until N) {
        
        // layer input
        for(i <- 0 until n_layers) {
          if(i == 0) {
            prev_layer_input = new Array[Double](n_ins)
            for(j <- 0 until n_ins) prev_layer_input(j) = train_X(n)(j)
          } else {
            prev_layer_input = new Array[Double](hidden_layer_sizes(i-1))
            for(j <- 0 until hidden_layer_sizes(i-1)) prev_layer_input(j) = layer_input(j)
          }

          layer_input = new Array[Double](hidden_layer_sizes(i))
          sigmoid_layers(i).sample_h_given_v(prev_layer_input, layer_input)
        }

        log_layer.train(layer_input, train_Y(n), lr)
      }
      // lr *= 0.95
    }
  }

  def predict(x: Array[Double], y: Array[Double]) {
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input: Array[Double] = new Array[Double](n_ins)

    var i: Int = 0
    var j: Int = 0
    var k: Int = 0

    for(j <- 0 until n_ins) prev_layer_input(j) = x(j)
    
    var linear_outoput: Double = 0

    // layer activation
    for(i <- 0 until n_layers) {
      layer_input = new Array[Double](sigmoid_layers(i).n_out)

      for(k <- 0 until sigmoid_layers(i).n_out) {
        linear_outoput = 0.0

        for(j <- 0 until sigmoid_layers(i).n_in) {
          linear_outoput += sigmoid_layers(i).W(k)(j) * prev_layer_input(j)
        }
        linear_outoput += sigmoid_layers(i).b(k)
        layer_input(k) = sigmoid(linear_outoput)
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


object DBN {
  def test_dbn() {
    val rng: Random = new Random(123)

    val pretrain_lr: Double = 0.5
    val pretraining_epochs: Int = 2000
    val k: Int = 1
    val finetune_lr: Double = 0.1
    val finetune_epochs: Int = 10000
    

    val hidden_layer_sizes: Array[Int] = Array(3, 3)
    val n_layers = hidden_layer_sizes.length
	
    val fea = scala.io.Source.fromFile("C:\\...\\iono_fea.csv")
    val train_X: Array[Array[Double]] = fea.getLines.map(line => {val tokens = line.split(","); tokens.map(_.toDouble)}).toArray
    fea.close()

    // training data
    //val train_X: Array[Array[Int]] = Array(
	//		Array(1, 1, 1, 0, 0, 0),
	//		Array(1, 0, 1, 0, 0, 0),
	//	    Array(1, 1, 1, 0, 0, 0),
	//		Array(0, 0, 1, 1, 1, 0),
	//		Array(0, 0, 1, 1, 0, 0),
	//		Array(0, 0, 1, 1, 1, 0)
    //)
	
	val tar = scala.io.Source.fromFile("C:\\...\\iono_tar.csv")
    val train_Y: Array[Array[Double]] = tar.getLines.map(line => {val tokens = line.split(","); tokens.map(_.toDouble)}).toArray
    tar.close()

    //val train_Y: Array[Array[Int]] = Array(
	//		Array(1, 0),
	//		Array(1, 0),
	//		Array(1, 0),
	//		Array(0, 1),
	//		Array(0, 1),
	//		Array(0, 1)
    //)
    
	
			// test data
	val test_X: Array[Array[Double]] = train_X
	//Array(
	//	Array(1, 1, 0, 0, 0, 0),
	//	Array(1, 1, 1, 1, 0, 0),
	//	Array(0, 0, 0, 1, 1, 0),
	//	Array(1, 1, 1, 1, 1, 0),
	//	Array(0, 0, 0, 0.4, 1, 0),
	//	Array(0, 0, 0.5, 1, 1, 0),
	//	Array(0, 0, 0.5, 1, 0.3, 0)
	//)
		
	
	val train_N: Int = train_X.length
	val n_outs: Int = 2
	val test_N: Int = test_X.length
    val n_ins: Int = train_X(0).length
	
	
    // construct DBN
    val dbn: DBN = new DBN(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, rng)

		// pretrain
		dbn.pretrain(train_X, pretrain_lr, k, pretraining_epochs);
		
		// finetune
		dbn.finetune(train_X, train_Y, finetune_lr, finetune_epochs);
	
    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_outs)

    var i: Int = 0
    var j: Int = 0

    // test
    for(i <- 0 until test_N) {
      dbn.predict(test_X(i), test_Y(i))
      for(j <- 0 until n_outs) {
        print(test_Y(i)(j) + " ")
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
    
  }
  

  def main(args: Array[String]) {
    test_dbn()
  }
}
