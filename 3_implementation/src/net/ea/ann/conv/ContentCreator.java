/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.io.Serializable;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.function.Function;
import net.ea.ann.raster.Size;

/**
 * This interface represents a utility to create convolutional layer as content.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ContentCreator extends Serializable, Cloneable {

	
	/**
	 * Creating content with content activation function, size, and filter.
	 * @param contentActivateRef activation function of content, which is often activation function related to convolutional pixel like ReLU function.
	 * @param size layer size.
	 * @param filter kernel filter.
	 * @return created content.
	 */
	Content newContent(Function contentActivateRef, Size size, Filter filter);

		
}
