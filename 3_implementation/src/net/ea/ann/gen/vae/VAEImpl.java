/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.rmi.RemoteException;
import java.util.List;
import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandard;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class is the default implementation of Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class VAEImpl extends VAEAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Predefined minimum value ins this context.
	 */
	protected static final double MIN_VALUE = (double)Float.MIN_VALUE;
	
	
	/**
	 * Name of flag to indicate whether X variance is fixed. Fixed variance is identity matrix.
	 */
	public static final String FIXED_VAR_FIELD = "vae_fixed_var";
	

	/**
	 * Default value of the flag to indicate whether X variance is fixed. Fixed variance is identity matrix.
	 */
	public static final boolean FIXED_VAR_DEFAULT = false;

	
	/**
	 * Name of flag to indicate whether X variance is adjusted.
	 */
	public static final String ADJUST_VAR_FIELD = "vae_adjust_var";
	

	/**
	 * Default value of the flag to indicate whether X variance is adjusted.
	 */
	public static final boolean ADJUST_VAR_DEFAULT = true;

	
//	/**
//	 * Name of flag to indicate whether squared root of X variance is stored.
//	 */
//	public static final String STORE_VAR_SQRT_FIELD = "vae_store_varsqrt";
	

	/**
	 * Default value of the flag to indicate whether squared root of X variance is stored.
	 */
	public static final boolean STORE_VAR_SQRT_DEFAULT = true;

	
	/**
	 * Name of the field that indicates whether variance of X is computed as a set of single diagonal variances.
	 */
	public static final String VARX_DIAGONAL_FIELD = "vae_var_diagonal";

	
	/**
	 * Flag to indicate whether variance of X is computed as a set of single diagonal variances.
	 * If this field is true, variance of X is computed as a vector of single diagonal variances,
	 * which is simple case when covariance matrix is diagonal matrix.
	 * If this field is false, variance of X is computed as a covariance matrix.
	 */
	public static final boolean VARX_DIAGONAL_DEFAULT = true;

	
	/**
	 * Internal encoder.
	 */
	protected NetworkStandardImpl encoder = null;
	
	
	/**
	 * Internal decoder.
	 */
	protected NetworkStandardImpl decoder = null;
	
	
	/**
	 * Z1 = Mean of original data X encoded
	 */
	protected NeuronStandard[] muX = null;
	
	
	/**
	 * Z2 = Variance of original data X encoded
	 */
	protected NeuronStandard[][] varX = null;
	
	
	/**
	 * Inverse of variance of original data X encoded. This variable is derived from the variance {@link #varX}.
	 */
	private NeuronValue[][] varXInverse = null;
	

	/**
	 * Squared root of variance of original data X encoded. This variable is derived from the variance {@link #varX}.
	 */
	private NeuronValue[][] varXSqrtTemp = null;

	
//	/**
//	 * Flag to indicate whether variance of X is computed as a set of single diagonal variances.
//	 * If this property is true, variance of X is computed as a vector of single diagonal variances,
//	 * which is simple case when covariance matrix is diagonal matrix.
//	 * If this property is false, variance of X is computed as a covariance matrix.
//	 */
//	protected boolean varXDiagonal = VARX_DIAGONAL_DEFAULT;

	
	/**
	 * Internal randomizer.
	 */
	protected Random learnRnd = new Random();
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public VAEImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
		
		this.config.put(FIXED_VAR_FIELD, FIXED_VAR_DEFAULT);
		this.config.put(ADJUST_VAR_FIELD, ADJUST_VAR_DEFAULT);
		this.config.put(VARX_DIAGONAL_FIELD, VARX_DIAGONAL_DEFAULT);
		
		GeneratorStandard.fillConfig(this.config);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public VAEImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 * @param neuronChannel neuron channel.
	 */
	public VAEImpl(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	@Override
	protected NetworkStandardImpl createEncoder() {
		GeneratorStandard<?> generator = new GeneratorStandard<Trainer>(neuronChannel, activateRef, idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
				NeuronValue error = calcEncodedError(outputNeuron);
				if (error != null)
					return error;
				else if (outputLayer != null)
					return outputLayer.newNeuronValue().zero();
				else if (realOutput != null)
					return realOutput.zero();
				else
					return null;
			}

		};
		generator.setParent(this);
		return generator;
	}


	@Override
	protected NetworkStandardImpl createDecoder() {
		GeneratorStandard<?> generator = new GeneratorStandard<Trainer>(neuronChannel, activateRef, idRef);
		generator.setParent(this);
		return generator;
	}
	
	
	/**
	 * Initialize with X dimension and Z dimension as well as hidden neurons.
	 * @param xDim X dimension.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode) {
		if (xDim <= 0 || zDim <= 0) return false;
		
		this.encoder = createEncoder();
		if(!this.encoder.initialize(xDim, isVarXDiagonal() ? 2*zDim : zDim*(zDim+1), nHiddenNeuronEncode)) return false;
		
		LayerStandard encodeLayer = this.encoder.getOutputLayer();
		this.muX = new NeuronStandard[zDim];
		for (int i = 0; i < zDim; i++) this.muX[i] = encodeLayer.get(i);
		
		this.varX = new NeuronStandard[zDim][];
		NeuronValue zero = encodeLayer.newNeuronValue().zero();
		NeuronStandard zeroNeuron = encodeLayer.newNeuron();
		zeroNeuron.setInput(zero);
		zeroNeuron.setOutput(zero);
		for (int i = 0; i < zDim; i++) {
			this.varX[i] = new NeuronStandard[zDim];
			for (int j = 0; j < zDim; j++) {
				if (isVarXDiagonal())
					this.varX[i][j] = i == j ? encodeLayer.get(zDim + i) : zeroNeuron;
				else
					this.varX[i][j] = encodeLayer.get(zDim + i*zDim + j);
			}
		}

		this.decoder = createDecoder();
		if(!this.decoder.initialize(zDim, xDim, nHiddenNeuronDecode)) return false;
		
		return true;
	}
	
	
	/**
	 * Initialize with X dimension and Z dimension as well as hidden neurons.
	 * @param xDim X dimension.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronEncode) {
		return initialize(xDim, zDim, nHiddenNeuronEncode,
			nHiddenNeuronEncode != null && nHiddenNeuronEncode.length > 0? reverse(nHiddenNeuronEncode) : null);
	}
	
	
	/**
	 * Initialize with X dimension and Z dimension.
	 * @param xDim X dimension.
	 * @param zDim Z dimension
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int xDim, int zDim) {
		return initialize(xDim, zDim, NetworkStandard.constructHiddenNeuronNumbers(xDim, zDim, getHiddenLayerMin()));
	}


	@Override
	public void reset() throws RemoteException {
		encoder = null;
		decoder = null;
		muX = null;
		varX = null;
		varXInverse = null;
		varXSqrtTemp = null;
	}
	
	
	@Override
	protected NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			for (Record record : sample) {
				if (record == null) continue;
				
				try {
					//Learning encoder.
					encoder.learn(record.input, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
				
				try {
					//Learning decoder.
					error = decoder.learn(randomizeDataZ(learnRnd), record.input, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "vae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.isBooleanValue(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}
		
		//Fix something
		adjustVarX();
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "vae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	@Override
	protected NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			//Learning encoder.
			encoder.learn(sample, lr, terminatedThreshold, 1);
			
			List<Record> decodeSample = Util.newList(0);
			for (Record record : sample) {
				if (record == null) continue;
				decodeSample.add(new Record(randomizeDataZ(learnRnd), record.input));
			}
			//Learning decoder.
			error = decoder.learn(decodeSample, lr, terminatedThreshold, 1);
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "vae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.isBooleanValue(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}
		
		//Fix something
		adjustVarX();
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "vae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	@Override
	public synchronized G generate() throws RemoteException {
		NeuronValue[] dataZ = randomizeDataZ(learnRnd);
		NeuronValue[] genX = generateByZ(dataZ);
		
		G g = new G();
		g.z = dataZ;
		g.xgen = genX;
		return g;
	}


	@Override
	public synchronized G generateBest() throws RemoteException {
		if (muX == null || muX.length == 0) return null;
		
		NeuronValue zero = muX[0].getOutput().zero();
		NeuronValue[] rNumbers = new NeuronValue[muX.length];
		for (int i = 0; i < muX.length; i++) rNumbers[i] = zero; 
		
		NeuronValue[] dataZ = makeDataZ(rNumbers);
		NeuronValue[] genX = generateByZ(dataZ);
		if (genX == null || genX.length == 0) return null;
		
		G g = new G();
		g.z = dataZ;
		g.xgen = genX;
		
		return g;
	}


	/**
	 * Generate X data.
	 * @param dataZ Z data is encoded data.
	 * @return generated values (X data).
	 */
	protected NeuronValue[] generateByZ(NeuronValue...dataZ) {
		if (!isValid()) return null;
		if (dataZ == null) return null;
		
		try {
			return decoder.evaluate(new Record(dataZ));
		} catch (Throwable e) {}
		
		return null;
	}
	

	/**
	 * Checking whether this VAE is valid.
	 * @return whether this VAE is valid.
	 */
	protected boolean isValid() {
		return (encoder != null && decoder != null && muX != null && varX != null);
	}
	
	
	/**
	 * Randomize Z data which is encoded data.
	 * @param rnd specific randomizer.
	 * @return Z data which is encoded data.
	 */
	protected NeuronValue[] randomizeDataZ(Random rnd) {
		if (muX == null || muX.length == 0) return null;
		
		NeuronValue[] rNumbers = new NeuronValue[muX.length];
		NeuronValue zero = muX[0].getOutput().zero();
		for (int i = 0; i < muX.length; i++) {
			rNumbers[i] = zero.valueOf(Util.randomGaussian(rnd));
		}
		
		return makeDataZ(rNumbers);
	}
	
	
	/**
	 * Making Z data from some numbers.
	 * @param rNumbers some numbers.
	 * @return Z data made from some numbers.
	 */
	private NeuronValue[] makeDataZ(NeuronValue[] rNumbers) {
		NeuronValue[][] varXSqrt = getUpdateVarXValueSqrt();
		NeuronValue[] muXValue = getMuXValue();
		if (varXSqrt == null) return muXValue; //Considering rNumbers is zero vector.

		NeuronValue[] dataZ = NeuronValue.multiply(varXSqrt, rNumbers);
		for (int i = 0; i < dataZ.length; i++) dataZ[i] = dataZ[i].add(muXValue[i]);
		
		return dataZ;
	}

	
	/**
	 * Calculate error of an encoded neuron.
	 * This code is the most important code to implement and combine Variational Autoencoders (VAE) with backpropagation algorithm.
	 * The equation to calculate KL-divergence given standard Gaussian distribution N(0, I) is available in the book
	 * Tutorial on Variational Autoencoders by Carl Doersch, Carnegie Mellon / UC Berkeley, page 9.
	 * The KL-divergence given n-dimension standard Gaussian distribution, which measures the difference between
	 * the distribution of X (mean muX and covariance matrix varX) and the standard Gaussian distribution, is specified as follows:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;KL(N(muX, varX) | N(0, I)) = 1/2(trace(varX) + muX^muX - n - log(det(varX)))<br>
	 * Actually, KL(N(muX, varX) | N(0, I)) is equivalent to the KL-divergence between the distribution of encoded Z distribution and decoded X distribution
	 * because Z is calculated from mean muX and covariance matrix varX as follows:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;Z = muX + sqrt(varX)*r where r is random vector by standard Gaussian distribution N(0, I).<br>
	 * The gradient of KL(N(muX, varX) | N(0, I)) is:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;gradient(KL(N(muX, varX) | N(0, I))) = 1/2(I + 2muX - V^(-1)<br>
	 * Where<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;derivative(trace(varX)) = I<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;derivative(muX^muX) = 2mu<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;derivative(log(det(varX))) = V^(-1)<br>
	 * Obviously, the descending direction of KL-divergence in stochastic gradient descend (SGD) algorithm is:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;d(KL(N(muX, varX) | N(0, I))) = -gradient(KL(N(muX, varX) | N(0, I))) = 1/2(-I - 2muX + V^(-1)<br>
	 * My contribution here is to combine Variational Autoencoders and backpropagation algorithm in incorporating KL-divergence and calculating error in backpropagation.
	 * Note that the loss function in VAE is:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;loss = ||X' - X||^2 + KL(N(muX, varX) | N(0, I))<br>
	 * Where X' is forwarded from Z by the decoding neural network. The error ||X' - X||^2 is minimized by backpropagation as usual but
	 * this method here is to minimized KL(N(muX, varX) | N(0, I)) by SGD algorithm.<br>
	 * Minimize this loss function is to maximize the following log-likelihood function:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;log-likelihood = log-likelihood(P(X|Z)) - KL(N(muX, varX) | N(0, I))<br>
	 * Maximizing log-likelihood(P(X|Z)) is the same to minimize the error ||X' - X||^2 as usual. Note that Z is calculated from muX and varX as follows:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;Z = muX + sqrt(varX)*r where r is random vector by standard Gaussian distribution N(0, I).
	 * @param neuron specific encoded neuron.
	 * @return error or loss of the encode neuron.
	 * @author Carl Doersch, Loc Nguyen
	 */
	protected NeuronValue calcEncodedError(NeuronStandard neuron) {
		NeuronValue derivative = GeneratorStandard.Neuron.derivative(neuron);

		boolean isMu = false;
		for (NeuronStandard nr : muX) {
			if (nr == neuron) {
				isMu = true;
				break;
			}
		}
		if (isMu) {
			//Calculate derivative of mean. The derivative(muX^muX) = 2mu, which causes that the error is -mu.
			NeuronValue out = neuron.getOutput();
			return out.negative().multiplyDerivative(derivative);
		}
		
		boolean isVar = false;
		int row = 0, column = 0;
		for (int i = 0; i < varX.length; i++) {
			for (int j = 0; j < varX[i].length; j++) {
				NeuronStandard nr = varX[i][j];
				if (nr == neuron) {
					isVar = true;
					row = i; column = j;
					break;
				}
			}
			if (isVar) break;
		}
		if (!isVar) return null;
		
		NeuronValue[][] varXValue = getVarXValue();
		if (varXValue.length == 0) return null;
		
		if (neuron == varX[0][0] || varXInverse == null) {
			//Calculate the inverse of covariance matrix which is derivative of logarithm of determinant of covariance matrix.
			//It means that derivative(log(det(varX))) = V^(-1)
			updateVarXInverse();
		}
		if (varXInverse == null) return null;
		
		NeuronValue encodedError = varXInverse[row][column];
		if (row == column) {
			//Derivative of trace of covariance matrix is identity, derivative(trace(varX)) = I
			encodedError.subtract(encodedError.unit());
		}
		return encodedError.multiply(0.5).multiplyDerivative(derivative);
	}

	
	/**
	 * Getting X mean.
	 * @return X mean.
	 */
	protected NeuronValue[] getMuXValue() {
		NeuronValue[] muEncodeValues = new NeuronValue[muX.length];
		for (int i = 0; i < muX.length; i++) muEncodeValues[i] = muX[i].getOutput();
		
		return muEncodeValues;
	}
	
	
	/**
	 * Getting X variance.
	 * @return X variance.
	 */
	protected NeuronValue[][] getVarXValue() {
		NeuronValue[][] varXValue = new NeuronValue[varX.length][];
		for (int i = 0; i < varX.length; i++) {
			varXValue[i] = new NeuronValue[varX[i].length]; 
			for (int j = 0; j < varX[i].length; j++) {
				varXValue[i][j] = varX[i][j].getOutput();
				
				if (i == j) {
					//Preventing non-positive variance.
					NeuronValue max = varXValue[i][i].max(varXValue[i][i].valueOf(MIN_VALUE));
					if (!varXValue[i][i].equals(max)) {
						varX[i][i].setOutput(max);
						varXValue[i][i] = max;
					}
				}
			}
		}
		
		return varXValue;
	}
	
	
	/**
	 * Getting squared root of X variance.
	 * @return squared root of X variance.
	 */
	private NeuronValue[][] getUpdateVarXValueSqrt() {
		if (varXSqrtTemp != null && STORE_VAR_SQRT_DEFAULT) return varXSqrtTemp;
		
		NeuronValue[][] varXValue = getVarXValue();
		if (varXValue == null || varXValue.length == 0 || varXValue[0] == null || varXValue[0].length == 0) return (varXSqrtTemp = null);
		
		varXSqrtTemp = varXValue[0][0].matrixSqrt(varXValue); //This calling is better than calling NeuronValue.matrixSqrt0(varXValue);
		return varXSqrtTemp;
	}
	
	
	/**
	 * Resetting X covariance matrix to be invertible.
	 * @param identity flag to indicate identity matrix.
	 * @return inverse of X covariance matrix.
	 */
	private NeuronValue[][] resetVarX(boolean identity) {
		varXSqrtTemp = null; //This code line is important.

		if (isFixedVar()) { //Fixed variance is identity matrix.
			int zDim = varX.length; 
			for (int i = 0; i < zDim; i++) {
				for (int j = 0; j < zDim; j++) {
					NeuronValue out = varX[i][j].getOutput();
					if (i == j)
						varX[i][j].setOutput(out.unit());
					else
						varX[i][j].setOutput(out.zero());
				}
			}
			varXInverse = getVarXValue();
			return varXInverse;
		}
			
		Random rnd = new Random();
		int zDim = varX.length; 
		for (int i = 0; i < zDim; i++) {
			for (int j = 0; j < zDim; j++) {
				NeuronValue out = varX[i][j].getOutput();
				if (identity) {
					if (i == j)
						varX[i][j].setOutput(out.unit());
					else
						varX[i][j].setOutput(out.zero());
				}
				else {
					double r = rnd.nextDouble();
					r = Math.max(MIN_VALUE, r); //To prevent zero element.
					if (i == j)
						varX[i][j].setOutput(out.valueOf(r));
					else
						varX[i][j].setOutput(out.zero());
				}
			}
		}

		if (identity)
			varXInverse = getVarXValue();
		else {
			try {
				NeuronValue[][] varXValue = getVarXValue();
				varXInverse = varXValue[0][0].matrixInverse(varXValue);
			} catch (Throwable e) {varXInverse = null;}
		}
		
		return varXInverse;
	}
	
	
	/**
	 * Updating X covariance matrix.
	 * @return inverse of X covariance matrix.
	 */
	private NeuronValue[][] updateVarXInverse() {
		varXSqrtTemp = null; //This code line is important.
		
		if (isFixedVar()) return resetVarX(true);
		
		try {
			NeuronValue[][] varXValue = getVarXValue();
			varXInverse = varXValue[0][0].matrixInverse(varXValue);
		} catch (Throwable e) {varXInverse = null;}
		
		if (varXInverse == null) resetVarX(false);
		return varXInverse;
	}
	
	
	/**
	 * Adjusting X covariance matrix.
	 * @return inverse of X covariance matrix.
	 */
	protected NeuronValue[][] adjustVarX() {
		varXSqrtTemp = null; //This code line is important.
		
		if (!isAdjustVar()) return varXInverse;
		if (isFixedVar()) return resetVarX(true);

		int zDim = varX.length;
		boolean update = false;
		for (int i = 0; i < zDim; i++) {
			for (int j = i; j < zDim; j++) {
				NeuronValue out1 = varX[i][j].getOutput();
				NeuronValue out2 = varX[j][i].getOutput();
				
				if (i == j) { //Prevent non-positive variance
					NeuronValue max = out1.max(out1.valueOf(MIN_VALUE));
					if (!out1.equals(max)) {
						varX[i][i].setOutput(max);
						update = true;
					}
				}
				else { //Making symmetric covariance matrix.
					if (!out1.equals(out2)) {
						NeuronValue out = out1.add(out2).multiply(0.5); //This is a average trick to make symmetry.
						varX[i][j].setOutput(out);
						varX[j][i].setOutput(out);
						update = true;
					}
				}
			} //End j
		} //End i

		if (!update && varXInverse != null) return varXInverse; 
		
		NeuronValue[][] varXValue = getVarXValue();
		try {
			varXInverse = varXValue[0][0].matrixInverse(varXValue);
		} catch (Throwable e) {varXInverse = null;}
		if (varXInverse == null) resetVarX(true);
		return varXInverse;
	}

	
	/**
	 * Checking whether X variance is fixed. Fixed variance is identity matrix.
	 * @return whether X variance is fixed.
	 */
	private boolean isFixedVar() {
		return config.getAsBoolean(FIXED_VAR_FIELD);
	}
	
	
	/**
	 * Checking whether X variance is adjusted.
	 * @return whether X variance is adjusted.
	 */
	private boolean isAdjustVar() {
		return config.getAsBoolean(ADJUST_VAR_FIELD);
	}

	
	/**
	 * Checking whether variance of X is computed as a set of single diagonal variances.
	 * @return true if variance of X is computed as a vector of single diagonal variances,
	 * which is simple case when covariance matrix is diagonal matrix.
	 * Otherwise, returning false if variance of X is computed as a covariance matrix.
	 */
	boolean isVarXDiagonal() {
		return config.getAsBoolean(VARX_DIAGONAL_FIELD);
	}
	
	
	/**
	 * Converting this VAE to text.
	 * @return text representing this VAE.
	 */
	public String toText() {
		StringBuffer buffer = new StringBuffer();
		if (encoder != null) {
			buffer.append("Encoder:\n");
			buffer.append(encoder.toText());
		}
		
		if (decoder != null) {
			if (encoder != null) buffer.append("\n\n");
			buffer.append("Decoder:\n");
			buffer.append(decoder.toString());
		}
		
		return buffer.toString();
	}


	@Override
	public void close() throws Exception {
		super.close();
		reset();
	}


}
