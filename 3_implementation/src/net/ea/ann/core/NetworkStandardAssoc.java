/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

import net.ea.ann.core.value.NeuronValue;

/**
 * This class is an associator of standard neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NetworkStandardAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal standard network.
	 */
	protected NetworkStandardAbstract network = null;
	
	
	/**
	 * Constructor with specified standard network.
	 * @param network specified standard network.
	 */
	public NetworkStandardAssoc(NetworkStandardAbstract network) {
		this.network = network;
	}

	
	/**
	 * Setting all values with specified value.
	 * @param value specified value.
	 * @return this association.
	 */
	public NetworkStandardAssoc setValues(double value) {
		List<LayerStandard> layers = network.getAllLayers();
		for (LayerStandard layer : layers) {
			NeuronValue nv = layer.newNeuronValue().valueOf(value);
			for (int i = 0; i < layer.size(); i++) {
				NeuronStandard neuron = layer.get(i);
				neuron.setInput(nv);
				neuron.setOutput(nv);
			}
		}
		
		return this;
	}

	
	/**
	 * Set all weights with specified weight value.
	 * @param weight specified weight value.
	 * @return this association.
	 */
	public NetworkStandardAssoc setWeights(double weight) {
		List<LayerStandard> layers = network.getAllLayers();
		for (LayerStandard layer : layers) {
			for (int i = 0; i < layer.size(); i++) {
				NeuronStandard neuron = layer.get(i);
				WeightedNeuron[] wns = neuron.getNextNeurons();
				if (wns == null || wns.length == 0) continue;
				
				for (WeightedNeuron wn : wns) {
					NeuronValue weightValue = wn.weight.value.toValue();
					wn.weight.value = weightValue.valueOf(weight).toWeightValue();
				}
			}
		}
		
		return this;
	}
	
	
	/**
	 * Initializing weights with random numbers in interval [0, 1).
	 * @return this association.
	 */
	public NetworkStandardAssoc setWeights() {
		Random rnd = new Random();
		return setWeights(rnd.nextDouble());
	}
	
	
	/**
	 * Evaluating network with specified number array.
	 * @param values specified number array.
	 * @return array as output.
	 */
	public NeuronValue[] evaluate(double...values) {
		LayerStandard inputLayer = network.getInputLayer();
		if (inputLayer == null) return null;
		
		int n = values != null && values.length > 0 ? values.length : 0;
		NeuronValue[] nva = n > 0 ? new NeuronValue[n] : new NeuronValue[] {};
		NeuronValue zero = inputLayer.newNeuronValue().zero();
		for (int i = 0; i < n; i++) nva[i] = zero.valueOf(values[i]);
		
		try {
			return network.evaluate(new Record(nva));
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}
	
	
	/**
	 * Evaluating network with one specified value.
	 * @param value specified value.
	 * @return array as output.
	 */
	public NeuronValue[] evaluateByOne(double value) {
		LayerStandard inputLayer = network.getInputLayer();
		if (inputLayer == null) return null;
		
		int n = inputLayer.size();
		double[] values = new double[n];
		for (int i = 0; i < n; i++) values[i] = value;
		return evaluate(values);
	}
	
	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	public void initParams(double v) {
		List<LayerStandard> layers = network.getAllLayers();
		for (LayerStandard layer : layers) new LayerStandardAssoc(layer).initParams(v);
	}


	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	void initParams(Random rnd) {
		List<LayerStandard> layers = network.getAllLayers();
		for (LayerStandard layer : layers) new LayerStandardAssoc(layer).initParams(rnd);
	}

	
	/**
	 * Initializing parameters.
	 */
	public void initParams() {
		initParams(new Random());
	}
	
	
}
