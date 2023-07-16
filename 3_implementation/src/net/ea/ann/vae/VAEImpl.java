/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.vae;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.ea.ann.Function;
import net.ea.ann.FunctionLogistic1;
import net.ea.ann.Id;
import net.ea.ann.Layer;
import net.ea.ann.NetworkAbstract;
import net.ea.ann.NetworkDoEventImpl;
import net.ea.ann.NetworkStandardImpl;
import net.ea.ann.Neuron;
import net.ea.ann.NeuronValue;
import net.ea.ann.NeuronValue1;
import net.ea.ann.Record;
import net.ea.ann.Util;
import net.ea.ann.NetworkDoEvent.Type;

/**
 * This class is an implementation of Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class VAEImpl extends NetworkAbstract implements VAE {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;

	
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
	protected Neuron[] muEncode = null;
	
	
	/**
	 * Z2 = Variance of original data X encoded
	 */
	protected Neuron[][] varEncode = null;
	
	
	/**
	 * Inverse of matrix of encoding variance.
	 */
	private NeuronValue[][] varEncodeInverse = null;
	
	
	/**
	 * Constructor with ID reference and activation reference.
	 */
	public VAEImpl(Id idRef, Function activateRef) {
		super(idRef);
		this.activateRef = activateRef != null? activateRef : new FunctionLogistic1();
	}

	
	/**
	 * Constructor with ID reference.
	 */
	public VAEImpl(Id idRef) {
		this(idRef, null);
	}

	
	/**
	 * Default constructor.
	 */
	public VAEImpl() {
		this(null, null);
	}

	
	/**
	 * Initialize with X dimension and Z dimension as well as hidden neurons.
	 * @param xDim X dimension.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoding hidden neurons.
	 * @param nHiddenNeuronDecode number of decoding hidden neurons.
	 */
	public void initialize(int xDim, int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode) {
		this.encoder = new NetworkStandardImpl(idRef, activateRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected NeuronValue calcError(Neuron neuron, NeuronValue output) {
				return calcEncodeError(neuron);
			}
			
		};
		this.encoder.initialize(xDim, zDim * (zDim + 1), nHiddenNeuronEncode);
		
		Layer encodeLayer = this.encoder.getOutputLayer();
		this.muEncode = new Neuron[zDim];
		for (int i = 0; i < zDim; i++) {
			this.muEncode[i] = encodeLayer.get(i);
		}
		
		this.varEncode = new Neuron[zDim][];
		for (int i = 0; i < zDim; i++) {
			this.varEncode[i] = new Neuron[zDim];
			for (int j = 0; j < zDim; j++) {
				this.varEncode[i][j] = encodeLayer.get(zDim + i*zDim + j);
			}
		}

		this.decoder = new NetworkStandardImpl(idRef, activateRef);
		this.decoder.initialize(zDim, xDim, nHiddenNeuronDecode);
		
		//Updating invertible encoding variance.
		updateVarEncodeInverse();
	}
	
	
	/**
	 * Initialize with X dimension and Z dimension as well as hidden neurons.
	 * @param xDim X dimension.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoding hidden neurons.
	 */
	public void initialize(int xDim, int zDim, int[] nHiddenNeuronEncode) {
		initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronEncode != null && nHiddenNeuronEncode.length > 0? reverse(nHiddenNeuronEncode) : null);
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
					NeuronValue[] rNumbers = new NeuronValue[muEncode.length];
					for (int i = 0; i < muEncode.length; i++) {
						rNumbers[i] = muEncode[0].getOutput().identity().multiply(rnd.nextGaussian());
					}
					
					NeuronValue[][] varEncodeValues = getVarEncodeValues();
					varEncodeValues = NeuronValue.matrixSqrt(varEncodeValues);
					NeuronValue[] encodeValues = NeuronValue.matrixMultiply(varEncodeValues, rNumbers);
					NeuronValue[] muEncodeValues = getMuEncodeValues();
					for (int i = 0; i < encodeValues.length; i++) encodeValues[i] = encodeValues[i].add(muEncodeValues[i]);
					
					Record decodeRecord = new Record();
					decodeRecord.input = encodeValues;
					decodeRecord.output = record.input;
					decoder.eval(decodeRecord, true);
				} catch (Throwable e) {Util.trace(e);}

				
				try {
					//Updating weights and biases of encoder.
					List<Layer> encoderBackbone = encoder.getBackbone();
					encoder.bpLearn(encoderBackbone, record.input, null, learningRate, terminatedThreshold, maxIteration);
				} catch (Throwable e) {Util.trace(e);}
				
				try {
					//Updating weights and biases of encoder.
					List<Layer> decoderBackbone = decoder.getBackbone();
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

	
	/**
	 * Calculate error of an encoding neuron.
	 * @param neuron specific encoding neuron.
	 * @return error or loss of the encode neuron.
	 */
	protected NeuronValue calcEncodeError(Neuron neuron) {
		NeuronValue out = neuron.getOutput();
		NeuronValue derivative = neuron.getActivateRef().derivative(out);
		
		boolean isMu = false;
		for (Neuron nr : muEncode) {
			if (nr == neuron) {
				isMu = true;
				break;
			}
		}
		if (isMu) return out.negative().multiplyDerivative(derivative);
		
		boolean isVar = false;
		int row = 0, column = 0;
		for (int i = 0; i < varEncode.length; i++) {
			for (int j = 0; j < varEncode[i].length; j++) {
				Neuron nr = varEncode[i][j];
				if (nr == neuron) {
					isVar = true;
					row = i; column = j;
					break;
				}
			}
			if (isVar) break;
		}
		if (!isVar)
			return null;
		
		NeuronValue[][] varEncodeValues = getVarEncodeValues();
		if (varEncodeValues.length == 0)
			return null;
		
		if (neuron == varEncode[0][0] || varEncodeInverse == null)
			updateVarEncodeInverse();
		NeuronValue error = varEncodeInverse[row][column];
		if (row == column) error.subtract(error.identity());
		return error.multiply(0.5).multiplyDerivative(derivative);
	}

	
	/**
	 * Getting encoding means.
	 * @return encoding means.
	 */
	private NeuronValue[] getMuEncodeValues() {
		NeuronValue[] muEncodeValues = new NeuronValue[muEncode.length];
		for (int i = 0; i < muEncode.length; i++) {
			muEncodeValues[i] = muEncode[i].getOutput();
		}
		
		return muEncodeValues;
	}
	
	
	/**
	 * Getting encoding variances.
	 * @return encoding variances.
	 */
	private NeuronValue[][] getVarEncodeValues() {
		NeuronValue[][] varEncodeValues = new NeuronValue[varEncode.length][];
		for (int i = 0; i < varEncode.length; i++) {
			varEncodeValues[i] = new NeuronValue[varEncode[i].length]; 
			for (int j = 0; j < varEncode[i].length; j++) {
				varEncodeValues[i][j] = varEncode[i][j].getOutput();
			}
		}
		
		return varEncodeValues;
	}
	
	
	/**
	 * Updating inverse of matrix of encoding variance.
	 * @return inverse of matrix of encoding variance.
	 */
	protected NeuronValue[][] updateVarEncodeInverse() {
		if (varEncode == null || varEncode.length == 0) return null;
		
		NeuronValue[][] varEncodeValues = getVarEncodeValues();
		try {
			varEncodeInverse = varEncodeValues[0][0].matrixInverse(varEncodeValues);
		} catch (Throwable e) {
			varEncodeInverse = null;
		}
		
		if (varEncodeInverse == null) {
			resetVarEncodeIdentity();
			varEncodeInverse = getVarEncodeValues();
		}
		
		return varEncodeInverse;
	}
	
	
	/**
	 * Resetting encoding variance as identity.
	 */
	private void resetVarEncodeIdentity() {
		int zDim = varEncode.length; 
		for (int i = 0; i < zDim; i++) {
			for (int j = 0; j < zDim; j++) {
				NeuronValue out = varEncode[i][j].getOutput();
				if (i == j)
					varEncode[i][j].setOutput(out.identity());
				else
					varEncode[i][j].setOutput(out.zero());
			}
		}

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
		try (VAEImpl vae = new VAEImpl()) {
			vae.initialize(4, 2, new int[] {3});
		
			System.out.println(vae.toString());
			
			Record record = new Record();
			record.input = new NeuronValue[] {new NeuronValue1(1), new NeuronValue1(2), new NeuronValue1(3)};
			record.output = null;
			vae.learn(Arrays.asList(record));
			
			System.out.println(vae.toString());
		}
		catch (Exception e) {
			
		}
	}


}
