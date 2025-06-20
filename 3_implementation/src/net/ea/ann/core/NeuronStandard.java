/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.util.List;
import java.util.Set;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;

/**
 * This interface represents standard neuron.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NeuronStandard extends Neuron {

	
	/**
	 * Flag of derivative ramp.
	 * True flag is applied to the trick to squash the derivative at output rather than at input.
	 */
	static final boolean DERIVATIVE_RAMP = false;
	
	
	/**
	 * Getting identifier of neuron.
	 * @return identifier of neuron.
	 */
	int id();
	
	
	/**
	 * Getting input value.
	 * @return input value.
	 */
	NeuronValue getInput();
	
	
	/**
	 * Setting input value.
	 * @param value input value.
	 */
	void setInput(NeuronValue value);
	
	
	/**
	 * Getting bias.
	 * @return bias.
	 */
	NeuronValue getBias();
	
	
	/**
	 * Setting bias.
	 * @param bias specified bias.
	 */
	void setBias(NeuronValue bias);

	
	/**
	 * Getting output value.
	 * @return output value.
	 */
	NeuronValue getOutput();
	
	
	/**
	 * Setting output value.
	 * @param value output value.
	 */
	void setOutput(NeuronValue value);
	
	
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
	 * Getting previous neurons.
	 * @return previous neurons.
	 */
	WeightedNeuron[] getPrevNeurons();

		
	/**
	 * Getting previous neurons.
	 * @param prevLayer previous layer.
	 * @return previous neurons.
	 */
	WeightedNeuron[] getPrevNeurons(LayerStandard prevLayer);

		
	/**
	 * Getting implicit previous neurons.
	 * Previous implicit neurons are neither memory neurons nor rib-in neurons because
	 * memory neurons and rib-in neurons can be included as explicit previous neurons.
	 * @return implicit previous neurons.
	 */
	WeightedNeuron[] getPrevNeuronsImplicit();

	
	/**
	 * Getting next neurons.
	 * @return next neurons.
	 */
	WeightedNeuron[] getNextNeurons();

		
	/**
	 * Getting next neurons.
	 * @param nextLayer next layer.
	 * @return next neurons.
	 */
	WeightedNeuron[] getNextNeurons(LayerStandard nextLayer);

	
	/**
	 * Adding next neuron along with weight.
	 * @param neuron next neuron.
	 * @param weight next weight.
	 * @return true if adding is successful.
	 */
	boolean setNextNeuron(NeuronStandard neuron, Weight weight);
	
	
	/**
	 * Removing next neuron.
	 * @param neuron next neuron.
	 * @return true if removing is successful.
	 */
	boolean removeNextNeuron(NeuronStandard neuron);

	
	/**
	 * Clearing next neurons.
	 */
	void clearNextNeurons();
	
	
	/**
	 * Finding next neuron.
	 * @param neuron specified next neuron.
	 * @return next neuron.
	 */
	WeightedNeuron findNextNeuron(NeuronStandard neuron);
	
	
	/**
	 * Finding next neuron by specified identifier.
	 * @param neuronId specified next neuron identifier.
	 * @return next neuron.
	 */
	WeightedNeuron findNextNeuron(int neuronId);

	
	/**
	 * Getting input rib neurons.
	 * @return input rib neurons.
	 */
	WeightedNeuron[] getRibinNeurons();

	
	/**
	 * Adding input rib neuron along with weight.
	 * @param ribinNeuron input rib neuron.
	 * @param weight input rib weight.
	 * @return true if adding is successful.
	 */
	boolean setRibinNeuron(NeuronStandard ribinNeuron, Weight weight);
	
	
	/**
	 * Removing input rib neuron.
	 * @param ribinNeuron input rib neuron.
	 * @return true if removing is successful.
	 */
	boolean removeRibinNeuron(NeuronStandard ribinNeuron);

	
	/**
	 * Clearing input rib neurons.
	 */
	void clearRibinNeurons();
	
	
	/**
	 * Finding input rib neuron.
	 * @param ribinNeuron specified input rib neuron.
	 * @return input rib neuron.
	 */
	WeightedNeuron findRibinNeuron(NeuronStandard ribinNeuron);

	
	/**
	 * Finding input rib neuron by specified identifier.
	 * @param ribinNeuronId specified input rib neuron identifier.
	 * @return input rib neuron.
	 */
	WeightedNeuron findRibinNeuron(int ribinNeuronId);

	
	/**
	 * Getting output rib neurons.
	 * @return output rib neurons.
	 */
	WeightedNeuron[] getRiboutNeurons();

	
	/**
	 * Adding output rib neuron along with weight.
	 * @param riboutNeuron output rib neuron.
	 * @param weight output rib weight.
	 * @return true if adding is successful.
	 */
	boolean setRiboutNeuron(NeuronStandard riboutNeuron, Weight weight);
	
	
	/**
	 * Removing output rib neuron.
	 * @param riboutNeuron output rib neuron.
	 * @return true if removing is successful.
	 */
	boolean removeRiboutNeuron(NeuronStandard riboutNeuron);

	
	/**
	 * Clearing output rib neurons.
	 */
	void clearRiboutNeurons();
	
	
	/**
	 * Finding output rib neuron.
	 * @param riboutNeuron specified output rib neuron.
	 * @return output rib neuron.
	 */
	WeightedNeuron findRiboutNeuron(NeuronStandard riboutNeuron);

	
	/**
	 * Finding output rib neuron by specified identifier.
	 * @param riboutNeuronId specified output rib neuron identifier.
	 * @return output rib neuron.
	 */
	WeightedNeuron findRiboutNeuron(int riboutNeuronId);

	
	/**
	 * Getting inside previous neurons.
	 * @return inside previous neurons.
	 */
	List<WeightedNeuron> getInsidePrevNeurons();

	
	/**
	 * Getting inside next neurons.
	 * @return inside next neurons.
	 */
	List<WeightedNeuron> getInsideNextNeurons();

	
	/**
	 * Getting outside previous neurons.
	 * @return outside previous neurons.
	 */
	Set<WeightedNeuron> getOutsidePrevNeurons();

	
	/**
	 * Finding outside previous neuron.
	 * @param outsidePrevNeuron specified outside previous neuron.
	 * @return found outside previous neuron.
	 */
	WeightedNeuron findOutsidePrevNeuron(NeuronStandard outsidePrevNeuron);
	
	
	/**
	 * Adding outside previous neuron.
	 * @param outsidePrevNeuron outside previous neuron.
	 * @param weight weight.
	 * @return true if adding is successful.
	 */
	boolean addOutsidePrevNeuron(NeuronStandard outsidePrevNeuron, Weight weight);
	
	
	/**
	 * Removing outside previous neuron.
	 * @param outsidePrevNeuron outside previous neuron.
	 * @return true if removal is successful..
	 */
	boolean removeOutsidePrevNeuron(NeuronStandard outsidePrevNeuron);
	
	
	/**
	 * Clearing outside previous neuron.
	 */
	void clearOutsidePrevNeurons();

	
	/**
	 * Getting outside next neurons.
	 * @return outside next neurons.
	 */
	Set<WeightedNeuron> getOutsideNextNeurons();

	
	/**
	 * Getting outside next neurons regarding specified layer.
	 * @param layer specified layer.
	 * @return outside next neurons regarding specified layer.
	 */
	Set<WeightedNeuron> getOutsideNextNeurons(LayerStandard layer);
	
	
	/**
	 * Finding outside next neuron.
	 * @param outsideNextNeuron specified outside previous neuron.
	 * @return found outside next neuron.
	 */
	WeightedNeuron findOutsideNextNeuron(NeuronStandard outsideNextNeuron);
	
	
	/**
	 * Adding outside next neuron.
	 * @param outsideNextNeuron outside previous neuron.
	 * @param weight weight.
	 * @return true if adding is successful.
	 */
	boolean addOutsideNextNeuron(NeuronStandard outsideNextNeuron, Weight weight);
	
	
	/**
	 * Removing outside next neuron.
	 * @param outsideNextNeuron outside previous neuron.
	 * @return true if removal is successful..
	 */
	boolean removeOutsideNextNeuron(NeuronStandard outsideNextNeuron);
	
	
	/**
	 * Clearing outside next neuron.
	 */
	void clearOutsideNextNeurons();

	
	/**
	 * Getting previous sibling neuron.
	 * @return previous sibling neuron.
	 */
	NeuronStandard getPrevSibling();
	
	
	/**
	 * Getting next sibling neuron.
	 * @return next sibling neuron.
	 */
	NeuronStandard getNextSibling();
	
	
	/**
	 * Getting main layer.
	 * @return main layer.
	 */
	LayerStandard getLayer();
	
	
	/**
	 * Evaluating neuron output.
	 * @return neuron output.
	 */
	NeuronValue evaluate();


	/**
	 * Taking derivative at this neuron (input or output).
	 * @return derivative at this neuron (input or output).
	 */
	NeuronValue derivative();


	/**
	 * Getting local value as input for derivative.
	 * @param neuron specified neuron.
	 * @return input for derivative.
	 */
	static NeuronValue getDerivativeInput(NeuronStandard neuron) {
		if (neuron == null) return null;
		NeuronValue value = DERIVATIVE_RAMP ? neuron.getOutput() : neuron.getInput();
		return value != null ? value : neuron.getOutput();
	}

}
