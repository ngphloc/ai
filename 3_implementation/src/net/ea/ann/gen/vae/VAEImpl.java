/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.ea.ann.Id;
import net.ea.ann.LayerStandard;
import net.ea.ann.NetworkDoEvent.Type;
import net.ea.ann.NetworkDoEventImpl;
import net.ea.ann.NetworkStandardImpl;
import net.ea.ann.Neuron;
import net.ea.ann.NeuronValue;
import net.ea.ann.NeuronValue1;
import net.ea.ann.Record;
import net.ea.ann.Util;
import net.ea.ann.function.Function;
import net.ea.ann.function.LogisticFunction1;

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
	protected Neuron[] muX = null;
	
	
	/**
	 * Z2 = Variance of original data X encoded
	 */
	protected Neuron[][] varX = null;
	
	
	/**
	 * Inverse of variance of original data X encoded.
	 */
	private NeuronValue[][] varXInverse = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public VAEImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
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
	 */
	public VAEImpl() {
		this(1, null, null);
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
		
		this.encoder = new NetworkStandardImpl(neuronChannel, activateRef, idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected NeuronValue calcError(Neuron neuron, NeuronValue output) {
				return calcEncodedError(neuron);
			}
			
		};
		this.encoder.initialize(xDim, zDim * (zDim + 1), nHiddenNeuronEncode);
		
		LayerStandard encodeLayer = this.encoder.getOutputLayer();
		this.muX = new Neuron[zDim];
		for (int i = 0; i < zDim; i++) {
			this.muX[i] = encodeLayer.get(i);
		}
		
		this.varX = new Neuron[zDim][];
		for (int i = 0; i < zDim; i++) {
			this.varX[i] = new Neuron[zDim];
			for (int j = 0; j < zDim; j++) {
				this.varX[i][j] = encodeLayer.get(zDim + i*zDim + j);
			}
		}

		this.decoder = new NetworkStandardImpl(neuronChannel, activateRef, idRef);
		this.decoder.initialize(zDim, xDim, nHiddenNeuronDecode);
		
		//Updating invertible encoded variance.
		updateVarXInverse();
		
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
	
	
	@Override
	public synchronized NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return bpLearn(sample, learningRate, terminatedThreshold, maxIteration);
	}


	/**
	 * Learning neural network by back propagate algorithm.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected synchronized NeuronValue[] bpLearn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		Random rnd = new Random();
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			for (Record record : sample) {
				if (record == null || record.input == null) continue;
				
				//Evaluating encoder.
				try {
					encoder.eval(record, true);
				} catch (Throwable e) {Util.trace(e);}
				
				//Evaluating decoder.
				try {
					Record decodeRecord = new Record();
					decodeRecord.input = randomizeDataZ(rnd);
					decodeRecord.output = record.input;
					decoder.eval(decodeRecord, true);
				} catch (Throwable e) {Util.trace(e);}

				
				try {
					//Updating weights and biases of encoder.
					List<LayerStandard> encoderBackbone = encoder.getBackbone();
					encoder.bpLearn(encoderBackbone, record.input, null, learningRate, terminatedThreshold, maxIteration);
				} catch (Throwable e) {Util.trace(e);}
				
				try {
					//Updating weights and biases of encoder.
					List<LayerStandard> decoderBackbone = decoder.getBackbone();
					error = decoder.bpLearn(decoderBackbone, encoder.getOutputLayer().getOutput(), record.input, learningRate, terminatedThreshold, maxIteration);
				} catch (Throwable e) {Util.trace(e);}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "ann_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0)
				doStarted = false;
			else {
				double errorMean = 0;
				for (NeuronValue r : error) errorMean += r.norm();
				errorMean = errorMean / error.length;
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
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "ann_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	
	@Override
	public NeuronValue[] generate() throws RemoteException {
		NeuronValue[] dataZ = randomizeDataZ(new Random());
		return generate(dataZ);
	}


	/**
	 * Generate X data.
	 * @param dataZ Z data is encoded data.
	 * @return generated values (X data).
	 */
	private NeuronValue[] generate(NeuronValue[] dataZ) {
		if (!isValid()) return null;
		if (dataZ == null || dataZ.length != muX.length) return null;
		
		Record record = new Record();
		record.input = dataZ;
		try {
			return decoder.eval(record, true);
		} catch (Throwable e) {}
		
		return null;
	}
	

	/**
	 * Checking whether this VAE is valid.
	 * @return whether this VAE is valid.
	 */
	public boolean isValid() {
		return (encoder != null && decoder != null && muX != null && varX != null);
	}
	
	
	/**
	 * Randomize Z data which is encoded data.
	 * @param rnd specific randomizer.
	 * @return Z data which is encoded data.
	 */
	protected NeuronValue[] randomizeDataZ(Random rnd) {
		rnd = rnd != null ? rnd : new Random();
		NeuronValue[] rNumbers = new NeuronValue[muX.length];
		for (int i = 0; i < muX.length; i++) {
			rNumbers[i] = muX[0].getOutput().identity().multiply(rnd.nextGaussian());
		}
		
		NeuronValue[][] varXValue = getVarXValue();
		varXValue = NeuronValue.matrixSqrt(varXValue);
		NeuronValue[] dataZ = NeuronValue.matrixMultiply(varXValue, rNumbers);
		NeuronValue[] muXValue = getMuXValue();
		for (int i = 0; i < dataZ.length; i++) dataZ[i] = dataZ[i].add(muXValue[i]);
		
		return dataZ;
	}
	
	
	/**
	 * Calculate error of an encoded neuron.
	 * @param neuron specific encoded neuron.
	 * @return error or loss of the encode neuron.
	 */
	protected NeuronValue calcEncodedError(Neuron neuron) {
		NeuronValue out = neuron.getOutput();
		NeuronValue derivative = neuron.getActivateRef().derivative(out);
		
		boolean isMu = false;
		for (Neuron nr : muX) {
			if (nr == neuron) {
				isMu = true;
				break;
			}
		}
		if (isMu) return out.negative().multiplyDerivative(derivative);
		
		boolean isVar = false;
		int row = 0, column = 0;
		for (int i = 0; i < varX.length; i++) {
			for (int j = 0; j < varX[i].length; j++) {
				Neuron nr = varX[i][j];
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
		
		if (neuron == varX[0][0] || varXInverse == null) updateVarXInverse();
		
		NeuronValue encodedError = varXInverse[row][column];
		if (row == column) encodedError.subtract(encodedError.identity());
		return encodedError.multiply(0.5).multiplyDerivative(derivative);
	}

	
	/**
	 * Getting X mean.
	 * @return X mean.
	 */
	private NeuronValue[] getMuXValue() {
		NeuronValue[] muEncodeValues = new NeuronValue[muX.length];
		for (int i = 0; i < muX.length; i++) {
			muEncodeValues[i] = muX[i].getOutput();
		}
		
		return muEncodeValues;
	}
	
	
	/**
	 * Getting X variance.
	 * @return X variance.
	 */
	private NeuronValue[][] getVarXValue() {
		NeuronValue[][] varXValue = new NeuronValue[varX.length][];
		for (int i = 0; i < varX.length; i++) {
			varXValue[i] = new NeuronValue[varX[i].length]; 
			for (int j = 0; j < varX[i].length; j++) {
				varXValue[i][j] = varX[i][j].getOutput();
			}
		}
		
		return varXValue;
	}
	
	
	/**
	 * Updating inverse of matrix of X variance.
	 * @return inverse of matrix of X variance.
	 */
	protected NeuronValue[][] updateVarXInverse() {
		if (varX == null || varX.length == 0) return null;
		
		NeuronValue[][] varXValue = getVarXValue();
		try {
			varXInverse = varXValue[0][0].matrixInverse(varXValue);
		} catch (Throwable e) {
			varXInverse = null;
		}
		
		if (varXInverse == null) {
			resetVarXIdentity();
			varXInverse = getVarXValue();
		}
		
		return varXInverse;
	}
	
	
	/**
	 * Resetting encoded variance as identity.
	 */
	private void resetVarXIdentity() {
		int zDim = varX.length; 
		for (int i = 0; i < zDim; i++) {
			for (int j = 0; j < zDim; j++) {
				NeuronValue out = varX[i][j].getOutput();
				if (i == j)
					varX[i][j].setOutput(out.identity());
				else
					varX[i][j].setOutput(out.zero());
			}
		}

	}
	
	
	/**
	 * Reversing an array.
	 * @param array specific array.
	 * @return reversed array.
	 */
	private static int[] reverse(int[] array) {
		if (array == null) return null;
		int[] r = new int[array.length];
		
		for (int i = 0; i < array.length; i++) r[i] = array[array.length - i - 1];
		
		return r;
	}
	
	
	@Override
	public String toString() {
		if (encoder == null || decoder == null) return super.toString();
		
		StringBuffer buffer = new StringBuffer();
		buffer.append("Encoder:\n");
		buffer.append(encoder.toString());
		
		buffer.append("\n\n");
		buffer.append("Decoder:\n");
		buffer.append(decoder.toString());
		
		return buffer.toString();
	}


	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		try (VAEImpl vae = new VAEImpl(1, new LogisticFunction1())) {
			vae.initialize(4, 2, new int[] {3});
		
			//System.out.println(vae.toString());
			
			Record record = new Record();
			record.input = new NeuronValue[] {new NeuronValue1(4), new NeuronValue1(3), new NeuronValue1(2), new NeuronValue1(1)};
			record.output = null;
			vae.learn(Arrays.asList(record));
			
			System.out.println(vae.toString());
			
			NeuronValue[] x = vae.generate();
			System.out.println(x);
		}
		catch (Exception e) {
			
		}
	}


}
