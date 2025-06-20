/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import net.ea.ann.conv.Content;
import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;

/**
 * This class is the default implementation of stack element layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ElementLayerImpl extends ElementLayerAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Constructor with neuron channel, stack, content, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param stack specified stack.
	 * @param content layer content.
	 * @param activateRef activation reference.
	 * @param idRef ID reference.
	 */
	protected ElementLayerImpl(int neuronChannel, Stack stack, Content content, Function activateRef, Id idRef) {
		super(neuronChannel, stack, content, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and stack.
	 * @param neuronChannel neuron channel or depth.
	 * @param stack specified stack.
	 * @param content layer content.
	 * @param activateRef activation reference.
	 */
	protected ElementLayerImpl(int neuronChannel, Stack stack, Content content, Function activateRef) {
		this(neuronChannel, stack, content, activateRef, null);
	}

	
	/**
	 * Creating elemental layer with neuron channel, stack, content, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param stack specified stack.
	 * @param content layer content.
	 * @param activateRef activation reference.
	 * @param idRef ID reference.
	 * @return elemental layer.
	 */
	public static ElementLayerImpl create(int neuronChannel, Stack stack, Content content, Function activateRef, Id idRef) {
		if (content == null) return null;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new ElementLayerImpl(neuronChannel, stack, content, activateRef, idRef);
	}


	/**
	 * Creating elemental layer with neuron channel, stack, content, and activation function.
	 * @param neuronChannel neuron channel or depth.
	 * @param stack specified stack.
	 * @param content layer content.
	 * @param activateRef activation reference.
	 * @return elemental layer.
	 */
	public static ElementLayerImpl create(int neuronChannel, Stack stack, Content content, Function activateRef) {
		return create(neuronChannel, stack, content, activateRef, null);
	}


}
