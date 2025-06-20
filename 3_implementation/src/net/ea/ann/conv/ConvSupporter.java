/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.core.value.NeuronValueCreator;

/**
 * This interface provides utility methods for convolutional tasks.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvSupporter extends Cloneable {

	
	/**
	 * Getting convolutional neuron value creator.
	 * @return convolutional neuron value creator.
	 */
	NeuronValueCreator getConvNeuronValueCreator();
	
	
	/**
	 * Getting filter factory.
	 * @return filter factory.
	 */
	FilterFactory getFilterFactory();
	

}
