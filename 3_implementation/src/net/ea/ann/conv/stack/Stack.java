/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.util.List;
import java.util.Set;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ContentCreator;
import net.ea.ann.conv.ConvLayer;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;
import net.ea.ann.raster.Size;

/**
 * This interface represents stack of convolutional layers.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Stack extends ConvLayer, ContentCreator {


	/**
	 * Creating layer with activation function, content activation function, size, and filter.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param size layer size.
	 * @param filter kernel filter. This filter can be null.
	 * @return created layer.
	 */
	ElementLayer newLayer(Function activateRef, Function contentActivateRef, Size size, Filter filter);
	
	
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
	 * Getting size of stack.
	 * @return size of stack.
	 */
	int size();
	
	
	/**
	 * Getting element layer at specified index.
	 * @param index specified index.
	 * @return element layer at specified index.
	 */
	ElementLayer get(int index);

	
	/**
	 * Adding an element layer to stack.
	 * @param layer specified element layer.
	 * @return true if adding is successful.
	 */
	boolean add(ElementLayer layer);

	
	/**
	 * Removing element layer at specified index.
	 * @param index specified index.
	 * @return removed element layer.
	 */
	ElementLayer remove(int index);

	
	/**
	 * Clearing stack.
	 */
	void clear();

	
	/**
	 * Finding specified layer.
	 * @param layer specified layer.
	 * @return index of specified layer.
	 */
	int indexOf(ElementLayer layer);
	
	
	/**
	 * Getting previous stack.
	 * @return previous stack.
	 */
	Stack getPrevStack();

	
	/**
	 * Getting all implicit and explicit previous stacks.
	 * @return all implicit and explicit previous stacks.
	 */
	Set<Stack> getAllPrevStacks();

	
	/**
	 * Getting next stack.
	 * @return next stack.
	 */
	Stack getNextStack();
	
	
	/**
	 * Setting next stack.
	 * @param nextStack next stack. It can be null.
	 * @return true if setting is successful.
	 */
	boolean setNextStack(Stack nextStack);


	/**
	 * Setting next stack.
	 * @param nextStack next stack.
	 * @param injective if this parameter is true, there is only one connection between two layers.
	 * @param filters array of filters.
	 * @return true if setting is successful.
	 */
	boolean setNextStack(Stack nextStack, boolean injective, Filter...filters);

		
	/**
	 * Setting content.
	 * @param datas array of content data.
	 * @return adjusted content data.
	 */
	NeuronValue[] setContent(NeuronValue[]...datas);

	
	/**
	 * Evaluating this stack.
	 * @return list of content data.
	 */
	List<Content> evaluate();
	
	
}
