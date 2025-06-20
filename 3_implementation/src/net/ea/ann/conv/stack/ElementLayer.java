/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.util.List;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ConvLayer;
import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Layer;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;

/**
 * This interface represents an element layer of layer stack.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ElementLayer extends Layer {

	
	/**
	 * Getting list of previous layers.
	 * @return list of previous layers.
	 */
	List<WeightedElementLayer> getPrevLayers();

	
	/**
	 * Getting list of next layers.
	 * @return list of next layers.
	 */
	List<WeightedElementLayer> getNextLayers();


	/**
	 * Getting list of next layers given next stack.
	 * @param nextStack next stack.
	 * @return list of next layers.
	 */
	List<WeightedElementLayer> getNextLayers(Stack nextStack);
	
	
	/**
	 * Setting next layer.
	 * @param layer next layer.
	 * @param weight next weight.
	 * @param filter next filter which can be null.
	 * @return true if setting is successful.
	 */
	boolean setNextLayer(ElementLayer layer, Weight weight, Filter filter);

	
	/**
	 * Removing next layer.
	 * @param layer next layer.
	 * @return true if removing is successful.
	 */
	boolean removeNextLayer(ElementLayer layer);

	
	/**
	 * Clearing next neurons.
	 */
	void clearNextLayers();

		
	/**
	 * Finding next elemental layer.
	 * @param layer specified next layer.
	 * @return next elemental layer.
	 */
	WeightedElementLayer findNextLayer(ElementLayer layer);
		

	/**
	 * Getting stack.
	 * @return current stack.
	 */
	Stack getStack();
	
	
	/**
	 * Getting content.
	 * @return layer content.
	 */
	Content getContent();
	
	
	/**
	 * Setting content.
	 * @param data content data.
	 * @return content data.
	 */
	NeuronValue[] setContent(NeuronValue[] data);
	
	
	/**
	 * Checking whether padding zero when filtering.
	 * @return whether padding zero when filtering.
	 */
	boolean isPadZeroFilter();
	
	
	/**
	 * Setting whether to pad zero when filtering.
	 * @param isPadZeroFilter flag to indicate whether to pad zero when filtering.
	 */
	void setPadZeroFilter(boolean isPadZeroFilter);
	
	
	/**
	 * Getting bias.
	 * @return bias of layer.
	 */
	NeuronValue getBias();
	
	
	/**
	 * Setting bias
	 * @param bias specified bias.
	 */
	void setBias(NeuronValue bias);
	
	
	/**
	 * Getting activation function.
	 * @return activation function.
	 */
	Function getActivateRef();

		
	/**
	 * Forwarding to evaluate the next layer.
	 * @return the next layer.
	 */
	ConvLayer forward();
	
	
	/**
	 * Evaluating this layer.
	 * @return convolutional layer as content.
	 */
	ConvLayerSingle evaluate();


}
