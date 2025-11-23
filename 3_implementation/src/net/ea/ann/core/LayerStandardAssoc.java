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
 * This class is an associator of standard layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LayerStandardAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal layer.
	 */
	protected LayerStandard layer = null;
	
	
	/**
	 * Constructor with specified layer.
	 * @param layer specified layer.
	 */
	public LayerStandardAssoc(LayerStandard layer) {
		this.layer = layer;
	}

	
	/**
	 * Getting standard layer.
	 * @return standard layer.
	 */
	private LayerStandardAbstract std() {
		return layer instanceof LayerStandardAbstract ? (LayerStandardAbstract)layer : null;
	}
	
	
	/**
	 * Checking whether layer is virtual layer.
	 * @return whether layer is virtual layer.
	 */
	public boolean isVirtualLayer() {
		return layer.getPrevLayer() == null && layer.getPrevLayerImplicit() == null &&
			layer.getNextLayer() == null &&
			layer.getRibinLayer() == null && layer.getRiboutLayer() == null;
	}
	
	
	/**
	 * Setting input rib layer.
	 * @param ribinLayer input rib layer. It can be null.
	 * @return true if setting is successful.
	 */
	public boolean setRibinLayer(LayerStandard ribinLayer) {
		return setRibinLayer(ribinLayer, false);
	}
	
	
	/**
	 * Setting input rib layer.
	 * @param ribinLayer input rib layer. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	protected boolean setRibinLayer(LayerStandard ribinLayer, boolean injective) {
		LayerStandardAbstract layer = std();
		if (layer == null) return false;
		return layer.setRibinLayer(ribinLayer, injective);
	}

	
	/**
	 * Setting output rib layer.
	 * @param riboutLayer output rib layer. It can be null.
	 * @return true if setting is successful.
	 */
	public boolean setRiboutLayer(LayerStandard riboutLayer) {
		return setRiboutLayer(riboutLayer, false);
	}
	
	
	/**
	 * Setting output rib layer.
	 * @param riboutLayer output rib layer. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	protected boolean setRiboutLayer(LayerStandard riboutLayer, boolean injective) {
		LayerStandardAbstract layer = std();
		if (layer == null) return false;
		return layer.setRiboutLayer(riboutLayer, injective);
	}

	
	/**
	 * Getting biases of this layer.
	 * @return biases of this layer.
	 */
	public NeuronValue[] getValues() {
		return layer.getOutput();
	}
	
	
	/**
	 * Getting regular weights of this layer, with regard to previous layer.
	 * @return regular weights of this layer, with regard to previous layer.
	 */
	public NeuronValue[][] getWeightsMy() {
		List<NeuronValue[]> weights = Util.newList(0);
		for (int  i = 0; i < layer.size(); i++) {
			if (!(layer.get(i) instanceof NeuronStandardImpl)) continue;
			NeuronStandardAssoc assoc = new NeuronStandardAssoc((NeuronStandardImpl)layer.get(i));
			weights.add(assoc.getWeightsMy());
		}
		return weights.toArray(new NeuronValue[][] {});
	}

	
	/**
	 * Getting implicit weights of this layer, with regard to previous layer.
	 * @return implicit weights of this layer, with regard to previous layer.
	 */
	public NeuronValue[][] getWeightsImplicitMy() {
		List<NeuronValue[]> weights = Util.newList(0);
		for (int  i = 0; i < layer.size(); i++) {
			if (!(layer.get(i) instanceof NeuronStandardImpl)) continue;
			NeuronStandardAssoc assoc = new NeuronStandardAssoc((NeuronStandardImpl)layer.get(i));
			weights.add(assoc.getWeightsImplicitMy());
		}
		return weights.toArray(new NeuronValue[][] {});
	}


	/**
	 * Getting all weights of this layer, with regard to previous layer.
	 * @return all weights of this layer, with regard to previous layer.
	 */
	public NeuronValue[][] getWeightsAllMy() {
		List<NeuronValue[]> weights = Util.newList(0);
		for (int  i = 0; i < layer.size(); i++) {
			if (!(layer.get(i) instanceof NeuronStandardImpl)) continue;
			NeuronStandardAssoc assoc = new NeuronStandardAssoc((NeuronStandardImpl)layer.get(i));
			weights.add(assoc.getWeightsAllMy());
		}
		return weights.toArray(new NeuronValue[][] {});
	}


	/**
	 * Getting regular weights with regard to previous layer.
	 * @return regular weights with regard to previous layer.
	 */
	public NeuronValue[][] getWeightsPrev() {
		return getWeightsMy();
	}


	/**
	 * Getting implicit weights with regard to previous layer.
	 * @return implicit weights with regard to previous layer.
	 */
	public NeuronValue[][] getWeightsImplicitPrev() {
		return getWeightsImplicitMy();
	}
	
	
	/**
	 * Getting all weights with regard to previous layer.
	 * @return all weights with regard to previous layer.
	 */
	public NeuronValue[][] getWeightsAllPrev() {
		return getWeightsAllMy();
	}
	
	
	/**
	 * Getting regular weights with regard to next layer.
	 * @return regular weights with regard to next layer.
	 */
	public NeuronValue[][] getWeightsNext() {
		LayerStandard nextLayer = layer.getNextLayer();
		return nextLayer != null ? new LayerStandardAssoc(nextLayer).getWeightsPrev() : new NeuronValue[][] {};
	}


	/**
	 * Getting implicit weights with regard to next layer.
	 * @return implicit weights with regard to next layer.
	 */
	public NeuronValue[][] getWeightsImplicitNext() {
		LayerStandard nextLayer = layer.getNextLayer();
		return nextLayer != null ? new LayerStandardAssoc(nextLayer).getWeightsImplicitPrev() : new NeuronValue[][] {};
	}
	
	
	/**
	 * Getting all weights with regard to next layer.
	 * @return all weights with regard to next layer.
	 */
	public NeuronValue[][] getWeightsAllNext() {
		LayerStandard nextLayer = layer.getNextLayer();
		return nextLayer != null ? new LayerStandardAssoc(nextLayer).getWeightsAllPrev() : new NeuronValue[][] {};
	}
	
	
	/**
	 * Getting biases of this layer.
	 * @return biases of this layer.
	 */
	public NeuronValue[] getBiases() {
		NeuronValue[] biases = new NeuronValue[layer.size()];
		for (int  i = 0; i < layer.size(); i++) {
			biases[i] = layer.get(i).getBias();
		}
		return biases;
	}


	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	public void initParams(double v) {
		LayerStandardAbstract std = std();
		for (NeuronStandard neuron : std.neurons) new NeuronStandardAssoc(neuron).initParams(v);
	}


	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	public void initParams(Random rnd) {
		LayerStandardAbstract std = std();
		for (NeuronStandard neuron : std.neurons) new NeuronStandardAssoc(neuron).initParams(rnd);
	}

	
}
