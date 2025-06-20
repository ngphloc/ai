/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.util.Set;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.core.value.Weight;

/**
 * This interface represents standard layer in standard neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface LayerStandard extends Layer, NeuronValueCreator {

	
	/**
	 * Create neuron.
	 * @return created neuron.
	 */
	NeuronStandard newNeuron();

	
	/**
	 * Create a new weight.
	 * @return new weight.
	 */
	Weight newWeight();

	
	/**
	 * Create bias.
	 * @return created bias.
	 */
	NeuronValue newBias();
	
	
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
	 * @return index of specified neuron.
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
	 * Getting implicit previous layer. By default, given a rib-out layer, its implicit previous layer is the layer on backbone to which it attaches.
	 * @return implicit previous layer.
	 */
	LayerStandard getPrevLayerImplicit();

	
	/**
	 * Checking whether having implicit or explicit previous layers.
	 * @return whether having implicit or explicit previous layers.
	 */
	boolean hasSomePrevLayers();
	
	
	/**
	 * Getting all implicit and explicit previous layers.
	 * @return all implicit and explicit previous layers.
	 */
	Set<LayerStandard> getAllPrevLayers();
	
	
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
	 * Checking whether having implicit or explicit next layers.
	 * @return whether having implicit or explicit next layers.
	 */
	boolean hasSomeNextLayers();
	
	
	/**
	 * Getting all implicit and explicit next layers.
	 * @return all implicit and explicit next layers.
	 */
	Set<LayerStandard> getAllNextLayers();

	
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
	 * Getting inside previous neurons.
	 * @return inside previous neurons.
	 */
	Set<WeightedNeuron> getInsidePrevNeurons();

	
	/**
	 * Getting inside previous virtual layer.
	 * @return inside previous virtual layer.
	 */
	LayerStandard getInsidePrevVirtualLayer();

	
	/**
	 * Getting inside next neurons.
	 * @return inside next neurons.
	 */
	Set<WeightedNeuron> getInsideNextNeurons();

	
	/**
	 * Getting inside next virtual layer.
	 * @return inside next virtual layer.
	 */
	LayerStandard getInsideNextVirtualLayer();

	
	/**
	 * Getting outside previous neurons.
	 * @return outside previous neurons.
	 */
	Set<WeightedNeuron> getOutsidePrevNeurons();

	
	/**
	 * Getting outside previous virtual layer.
	 * @return outside previous virtual layer.
	 */
	LayerStandard getOutsidePrevVirtualLayer();

	
	/**
	 * Setting outside previous virtual layer.
	 * @param outsidePrevVirtualLayer outside previous virtual layer.
	 * @return true if setting is successful.
	 */
	boolean addOutsidePrevVirtualLayer(LayerStandard outsidePrevVirtualLayer);

	
	/**
	 * Removing outside previous virtual layer.
	 * @param outsidePrevVirtualLayer outside previous virtual layer.
	 */
	void removeOutsidePrevVirtualLayer(LayerStandard outsidePrevVirtualLayer);

	
	/**
	 * Getting outside next neurons.
	 * @return outside next neurons.
	 */
	Set<WeightedNeuron> getOutsideNextNeurons();

	
	/**
	 * Getting outside next virtual layer.
	 * @return outside next virtual layer.
	 */
	LayerStandard getOutsideNextVirtualLayer();
	
	
	/**
	 * Setting outside next virtual layer.
	 * @param outsideNextVirtualLayer outside next virtual layer.
	 * @return true if setting is successful.
	 */
	boolean addOutsideNextVirtualLayer(LayerStandard outsideNextVirtualLayer);

	
	/**
	 * Removing outside next virtual layer.
	 * @param outsideNextVirtualLayer outside next virtual layer.
	 */
	void removeOutsideNextVirtualLayer(LayerStandard outsideNextVirtualLayer);

	
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
	 * Setting input values.
	 * @param input input values.
	 */
	void setInput(NeuronValue...input);
	
	
	/**
	 * Getting output values.
	 * @return output values.
	 */
	NeuronValue[] getOutput();
	
	
	/**
	 * Setting output values.
	 * @param output output values.
	 */
	void setOutput(NeuronValue...output);
	
	
	/**
	 * Evaluating a layer with specified input.
	 * @param input specified input.
	 * @return evaluated output.
	 */
	NeuronValue[] evaluate(NeuronValue[] input);

		
	/**
	 * Evaluating this layer.
	 * @return evaluated output.
	 */
	NeuronValue[] evaluate();
	
	
}
