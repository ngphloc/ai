/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ContentImpl;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Size;

/**
 * This class is the default implementation of stack of convolutional layers. It is a convolutional layer too.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class StackImpl extends StackAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Constructor with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 */
	protected StackImpl(int neuronChannel, Id idRef) {
		super(neuronChannel, idRef);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected StackImpl(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	/**
	 * Getting this stack.
	 * @return this stack.
	 */
	private StackImpl getThisStack() {return this;}
	
	
	@Override
	public NeuronValue newNeuronValue() {
		return NeuronValueCreator.newNeuronValue(neuronChannel);
	}

	
	@Override
	public Content newContent(Function contentActivateRef, Size size, Filter filter) {
		return new ContentImpl(neuronChannel, contentActivateRef, size, filter, idRef) {
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public NeuronValue newNeuronValue() {
				return getThisStack().newNeuronValue();
			}
			
		};
	}


	/**
	 * Creating content with content activation function and size.
	 * @param contentActivateRef activation function of content, which is often activation function related to convolutional pixel like ReLU function.
	 * @param size layer size.
	 * @return created content.
	 */
	public Content newContent(Function contentActivateRef, Size size) {
		return newContent(contentActivateRef, size, null);
	}
	
	
	@Override
	public ElementLayer newLayer(Function activateRef, Function contentActivateRef, Size size, Filter filter) {
		Content content = newContent(contentActivateRef, size, filter);
		return ElementLayerImpl.create(neuronChannel, this, content, activateRef, idRef);
	}

	
	/**
	 * Creating layer with activation function, content activation function, and size.
	 * @param activateRef activation function.
	 * @param contentActivateRef activation function of content.
	 * @param size layer size.
	 * @return created layer.
	 */
	public ElementLayer newLayer(Function activateRef, Function contentActivateRef, Size size) {
		return newLayer(activateRef, contentActivateRef, size, null);
	}
	
	
	/**
	 * Creating stack with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 * @return created stack.
	 */
	public static StackImpl create(int neuronChannel, Id idRef) {
		neuronChannel = neuronChannel < 1 ? 1: neuronChannel;
		return new StackImpl(neuronChannel, idRef);
	}

	
	/**
	 * Creating stack with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @return created stack.
	 */
	public static StackImpl create(int neuronChannel) {
		return create(neuronChannel, null);
	}


}
