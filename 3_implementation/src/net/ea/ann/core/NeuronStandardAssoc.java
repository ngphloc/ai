/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;

/**
 * This class is an associator of standard neuron.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronStandardAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Associative standard neuron.
	 */
	protected NeuronStandard neuron = null;
	
	
	/**
	 * Constructor with associative neuron.
	 * @param neuron associative neuron.
	 */
	public NeuronStandardAssoc(NeuronStandard neuron) {
		this.neuron = neuron;
	}

	
	/**
	 * Getting standard neuron.
	 * @return standard neuron.
	 */
	private NeuronStandardImpl std() {
		return neuron instanceof NeuronStandardImpl ? (NeuronStandardImpl)neuron : null;
	}
	
	
	/**
	 * Getting all weighted source neurons.
	 * @return all weighted source neurons.
	 */
	public List<WeightedNeuron> getSources() {
		return std().getSources();
	}

	
	/**
	 * Setting next neurons with array of weights.
	 * @param weights array of weights.
	 */
	public void setNextNeurons(Weight...weights) {
		if (weights == null || weights.length == 0) return;
		NeuronStandardImpl neuron = std();
		if (neuron == null) return;
		if (neuron.layer == null || neuron.layer.getNextLayer() == null) return;
		
		LayerStandard nextLayer = neuron.layer.getNextLayer();
		int n = Math.min(weights.length, nextLayer.size());
		for (int i = 0; i < n; i++) neuron.setNextNeuron(nextLayer.get(i), weights[i]);
	}
	
	
	/**
	 * Setting next rib-in neurons with array of weights.
	 * @param weights array of weights.
	 */
	public void setRibinNeurons(Weight...weights) {
		if (weights == null || weights.length == 0) return;
		NeuronStandardImpl neuron = std();
		if (neuron == null) return;
		if (neuron.layer == null || neuron.layer.getRibinLayer() == null) return;

		LayerStandard ribinLayer = neuron.layer.getRibinLayer();
		int n = Math.min(weights.length, ribinLayer.size());
		for (int i = 0; i < n; i++) neuron.setRibinNeuron(ribinLayer.get(i), weights[i]);
	}


	/**
	 * Setting next rib-out neurons with array of weights.
	 * @param weights array of weights.
	 */
	public void setRiboutNeurons(Weight...weights) {
		if (weights == null || weights.length == 0) return;
		NeuronStandardImpl neuron = std();
		if (neuron == null) return;
		if (neuron.layer == null || neuron.layer.getRiboutLayer() == null) return;

		LayerStandard riboutLayer = neuron.layer.getRiboutLayer();
		int n = Math.min(weights.length, riboutLayer.size());
		for (int i = 0; i < n; i++) neuron.setRiboutNeuron(riboutLayer.get(i), weights[i]);
	}


	/**
	 * Getting regular weights of this neuron, with regard to previous neurons.
	 * @return regular weights of this neuron, with regard to previous neurons.
	 */
	public NeuronValue[] getWeightsMy() {
		WeightedNeuron[] prevWeightedNeurons = neuron.getPrevNeurons();
		List<NeuronValue> weights = Util.newList(0);
		if (prevWeightedNeurons != null) {
			for (WeightedNeuron wn : prevWeightedNeurons) weights.add(wn.weight.value.toValue());
		}
		return weights.toArray(new NeuronValue[] {});
	}
	
	
	/**
	 * Getting implicit weights of this neuron, with regard to previous neurons.
	 * @return implicit weights of this neuron, with regard to previous neurons.
	 */
	public NeuronValue[] getWeightsImplicitMy() {
		WeightedNeuron[] prevWeightedNeuronsImplicit = neuron.getPrevNeuronsImplicit();
		List<NeuronValue> weights = Util.newList(0);
		if (prevWeightedNeuronsImplicit != null) {
			for (WeightedNeuron wn : prevWeightedNeuronsImplicit) weights.add(wn.weight.value.toValue());
		}
		return weights.toArray(new NeuronValue[] {});
	}


	/**
	 * Getting all weights of this neuron, with regard to previous neurons.
	 * @return all weights of this neuron, with regard to previous neurons.
	 */
	public NeuronValue[] getWeightsAllMy() {
		WeightedNeuron[] prevWeightedNeurons = neuron.getPrevNeurons();
		WeightedNeuron[] prevWeightedNeuronsImplicit = neuron.getPrevNeuronsImplicit();
		List<NeuronValue> weights = Util.newList(0);
		if (prevWeightedNeurons != null) {
			for (WeightedNeuron wn : prevWeightedNeurons) weights.add(wn.weight.value.toValue());
		}
		if (prevWeightedNeuronsImplicit != null) {
			for (WeightedNeuron wn : prevWeightedNeuronsImplicit) weights.add(wn.weight.value.toValue());
		}
		return weights.toArray(new NeuronValue[] {});
	}

	
	/**
	 * Getting previous neurons including outside previous neurons but excluding implicit previous neurons.
	 * @return previous neurons including outside previous neurons but excluding implicit previous neurons.
	 */
	public WeightedNeuron[] getPrevNeuronsIncludeOutside() {
		List<WeightedNeuron> prevNeuronList = Util.newList(0);
		prevNeuronList.addAll(Arrays.asList(neuron.getPrevNeurons()));
		prevNeuronList.addAll(neuron.getOutsidePrevNeurons());
		return prevNeuronList.toArray(new WeightedNeuron[] {});
	}

	
	/**
	 * Getting list of all weighted neurons.
	 * @return list of all weighted neurons.
	 */
	private List<WeightedNeuron> getAllWeightedNeurons() {
		List<WeightedNeuron> all = Util.newList(0);
		NeuronStandardImpl std = std();
		all.addAll(std.nextNeurons);
		all.addAll(std.riboutNeurons);
		all.addAll(std.insideNextNeurons);
		all.addAll(std.outsidePrevNeurons);
		all.addAll(std.outsideNextNeurons);
		return all;
	}
	
	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	public void initParams(double v) {
		NeuronStandardImpl std = std();
		if (std.bias != null) std.bias = std.bias.valueOf(v);
		List<WeightedNeuron> all = getAllWeightedNeurons();
		for (WeightedNeuron neuron : all) {
			neuron.weight.value = neuron.weight.value.toValue().valueOf(v).toWeightValue();
		}
	}


	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	public void initParams(Random rnd) {
		NeuronStandardImpl std = std();
		if (std.bias != null) std.bias = std.bias.valueOf(NeuronValue.r(rnd));
		List<WeightedNeuron> all = getAllWeightedNeurons();
		for (WeightedNeuron neuron : all) {
			neuron.weight.value = neuron.weight.value.toValue().valueOf(NeuronValue.r(rnd)).toWeightValue();
		}
	}

	
}
