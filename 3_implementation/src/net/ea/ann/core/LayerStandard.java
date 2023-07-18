/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import net.ea.ann.core.function.Function;

/**
 * This interface represents standard layer in standard neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface LayerStandard extends Layer {

	
	/**
	 * Creating an empty neuron value.
	 * @return empty neuron value.
	 */
	NeuronValue newNeuronValue();

	
	/**
	 * Create neuron.
	 * @return created neuron.
	 */
	NeuronStandard newNeuron();

	
	/**
	 * Getting layer size.
	 * @return layer size.
	 */
	int size();
	
	
	/**
	 * Getting neuron at specified index.
	 * @param index specified index.
	 * @return neuron at specified index.
	 */
	NeuronStandard get(int index);
	
	
	/**
	 * Adding neuron.
	 * @param neuron specified neuron.
	 * @return true if adding is successful.
	 */
	boolean add(NeuronStandard neuron);

	
	/**
	 * Removing neuron at specified index.
	 * @param index specified index.
	 * @return previous neuron.
	 */
	NeuronStandard remove(int index);
	
	
	/**
	 * Clearing all neurons.
	 */
	void clear();
	
	
	/**
	 * Finding specified neuron.
	 * @param neuron specified neuron.
	 * @return specified neuron.
	 */
	int indexOf(NeuronStandard neuron);
	
	
	/**
	 * Finding neuron by specified identifier.
	 * @param neuronId specified identifier.
	 * @return found neuron.
	 */
	int indexOf(int neuronId);
	
	
	/**
	 * Getting previous layer.
	 * @return previous layer.
	 */
	LayerStandard getPrevLayer();
	
	
	/**
	 * Getting implicit previous layer. For example, given a rib-out layer, its implicit previous layer is the layer on backbone to which it attaches.
	 * @return implicit previous layer.
	 */
	LayerStandard getPrevLayerImplicit();

	
	/**
	 * Setting previous layer.
	 * @param prevLayer previous layer. It can be null.
	 * @return true if setting is successful.
	 */
	boolean setPrevLayer(LayerStandard prevLayer);

	
	/**
	 * Getting next layer.
	 * @return next layer.
	 */
	LayerStandard getNextLayer();
	
	
	/**
	 * Setting next layer.
	 * @param nextLayer next layer. It can be null.
	 * @return true if setting is successful.
	 */
	boolean setNextLayer(LayerStandard nextLayer);
	
	
	/**
	 * Getting input rib layer.
	 * @return input rib layer.
	 */
	LayerStandard getRibinLayer();
	
	
	/**
	 * Setting input rib layer.
	 * @param ribinLayer input rib layer. It can be null.
	 * @return true if setting is successful.
	 */
	boolean setRibinLayer(LayerStandard ribinLayer);

	
	/**
	 * Getting output rib layer.
	 * @return output rib layer.
	 */
	LayerStandard getRiboutLayer();
	
	
	/**
	 * Setting output rib layer.
	 * @param riboutLayer output rib layer. It can be null.
	 * @return true if setting is successful.
	 */
	boolean setRiboutLayer(LayerStandard riboutLayer);
	
	
	/**
	 * Getting reference to activation function.
	 * @return reference to activation function.
	 */
	Function getActivateRef();
	
	
	/**
	 * Setting reference to activation function.
	 * @param activateRef reference to activation function.
	 * @return previous function reference.
	 */
	Function setActivateRef(Function activateRef);
	
	
	/**
	 * Getting input values.
	 * @return input values.
	 */
	NeuronValue[] getInput();
	
	
	/**
	 * Getting output values.
	 * @return output values.
	 */
	NeuronValue[] getOutput();
	
	
}
