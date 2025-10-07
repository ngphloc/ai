/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.rmi.RemoteException;
import java.util.Arrays;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NormSupporter;
import net.ea.ann.core.Record;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Raster2D;

/**
 * This class is the abstract implementation of generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class GenModelAbstract extends NetworkAbstract implements GenModel, NormSupporter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	protected GenModelAbstract(int neuronChannel, Function activateRef, Id idRef) {
		super(idRef);
		this.config.put(LEARN_MAX_ITERATION_FIELD, LEARN_MAX_ITERATION_DEFAULT);
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		this.config.put(Image.ALPHA_FIELD, Image.ALPHA_DEFAULT);
		this.config.put(Raster2D.LEARN_FIELD, Raster2D.LEARN_DEFAULT);
		this.config.put(HIDDEN_LAYER_MIN_FILED, HIDDEN_LAYER_MIN_DEFAULT);
		
		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
		this.activateRef = activateRef == null ? (activateRef = Raster.toActivationRef(this.neuronChannel, isNorm())) : activateRef;
	}
	

	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	protected GenModelAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected GenModelAbstract(int neuronChannel) {
		this(neuronChannel, null, null);
	}

	
	@Override
	public int getNeuronChannel() throws RemoteException {
		return neuronChannel;
	}


	/**
	 * Getting activation function.
	 * @return activation function.
	 */
	public Function getActivateRef() {
		return activateRef;
	}
	
	
	@Override
	public NeuronValue[] learnOneByOne(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = paramGetLearningRate();
		return learnOneByOne(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = paramGetLearningRate();
		return learn(sample, learningRate, terminatedThreshold, maxIteration);
	}


	/**
	 * Learning neural network, one-by-one record over sample.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected abstract NeuronValue[] learnOneByOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration);

		
	/**
	 * Learning neural network by back propagate algorithm.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected abstract NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration);

	
	@Override
	public synchronized G recover(NeuronValue[] dataX, Cube region, boolean random, boolean calcError) throws RemoteException {
		if (dataX == null || dataX.length == 0) return null;
		
		G g = random ? generate() : generateBest();
		if (g == null || g.xgen == null || g.xgen.length == 0) return null; 
		
		boolean entire = true;
		if (region != null && region.x >= 0) {
			region.width = region.x + region.width <= dataX.length ? region.width : dataX.length - region.x;
			if (region.width > 0) entire = false;
		}
		if (!entire) {
			for (int i = 0; i < g.xgen.length; i++) {
				if (i < region.x && i >= region.x + region.width) g.xgen[i] = dataX[i];
			}
		}
		
		double error = 0;
		int n = 0;
		if (calcError) {
			for (int i = 0; i < g.xgen.length; i++) {
				if (!entire && i < region.x && i >= region.x + region.width) continue;
				double d = g.xgen[i].subtract(dataX[i]).norm();
				error += Math.abs(d*(1-d));
				n++;
			}
		}
		
		g.error = n != 0 ? error/(double)n : 0;
		g.x = dataX;
		return g;
	}


	@Override
	public synchronized G reproduce(NeuronValue[] dataX, Cube region, boolean random, boolean calcError) throws RemoteException {
		learnOneByOne(Arrays.asList(new Record(dataX)));
		return recover(dataX, region, random, calcError);
	}


	@Override
	public boolean isNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}


	/**
	 * Setting normalization flag.
	 * @param isNorm normalization flag.
	 * @return this model.
	 */
	public GenModelAbstract setNorm(boolean isNorm) {
		if (this.isNorm() == isNorm) return this;
		config.put(Raster.NORM_FIELD, isNorm());
		activateRef = Raster.toActivationRef(neuronChannel, isNorm);
		return this;
	}
	
	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	protected int getDefaultAlpha() {
		if (config.containsKey(Image.ALPHA_FIELD))
			return config.getAsInt(Image.ALPHA_FIELD);
		else
			return Image.ALPHA_DEFAULT;
	}
	

	/**
	 * Getting minimum number of hidden layers.
	 * @return minimum number of hidden layers.
	 */
	protected int getHiddenLayerMin() {
		if (!config.containsKey(HIDDEN_LAYER_MIN_FILED)) return HIDDEN_LAYER_MIN_DEFAULT;
		int hiddenMin = config.getAsInt(HIDDEN_LAYER_MIN_FILED);
		return hiddenMin < HIDDEN_LAYER_MIN_DEFAULT ? HIDDEN_LAYER_MIN_DEFAULT : hiddenMin;
	}

	
	/**
	 * Reversing an array.
	 * @param array specific array.
	 * @return reversed array.
	 */
	protected static int[] reverse(int[] array) {
		if (array == null) return null;
		int[] r = new int[array.length];
		for (int i = 0; i < array.length; i++) r[i] = array[array.length - i - 1];
		return r;
	}
	
	
}
