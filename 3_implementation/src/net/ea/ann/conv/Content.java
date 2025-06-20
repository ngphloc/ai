/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.WeightValue;
import net.ea.ann.raster.NeuronRaster;
import net.ea.ann.raster.Size;

/**
 * This interface represents content of element layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Content extends ConvLayerSingle4D {


	/**
	 * Getting size.
	 * @return size of this content.
	 */
	Size getSize();

		
	/**
	 * Creating a new content by size.
	 * @param newSize specified size.
	 * @return new content.
	 */
	ContentImpl newContent(Size newSize);

	
	/**
	 * Creating a new content from data and bias.
	 * @param data specified data.
	 * @param bias specified bias.
	 * @return new content.
	 */
	Content newContent(NeuronValue[] data, NeuronValue bias);

		
	/**
	 * Duplicating this content.
	 * @return duplicated content.
	 */
	Content duplicateContent();
	
	
	/**
	 * Resizing content with new size.
	 * @param newSize new size.
	 * @return resized content.
	 */
	Content resizeContent(Size newSize);
	
	
	/**
	 * Adding this content with other content.
	 * @param content other content
	 * @return added content.
	 */
	Content add(Content content);
	
	
	/**
	 * Subtracting this content from other content.
	 * @param content other content
	 * @return subtracted content.
	 */
	Content subtract(Content content);

	
	/**
	 * Multiplying this content with other content.
	 * @param content other content
	 * @return multiplied content.
	 */
	Content multiply(Content content);

	
	/**
	 * Multiplying this content with weight.
	 * @param weight other weight
	 * @return multiplied content.
	 */
	Content multiply0(WeightValue weight);

	
	/**
	 * Multiplying this content with derivative.
	 * @param content other content
	 * @return multiplied content.
	 */
	Content multiplyDerivative(Content content);

	
	/**
	 * Multiplying this content with a real number.
	 * @param value a real number
	 * @return multiplied content.
	 */
	Content multiply0(double value);

	
	/**
	 * Dividing this content bya real number.
	 * @param value a real number
	 * @return multiplied content.
	 */
	Content divide0(double value);

	
	/**
	 * Calculating mean of this content.
	 * @return mean of this content.
	 */
	NeuronValue mean0();
	
	
	/**
	 * Calculating derivative of this content.
	 * @param f function.
	 * @return the content that was taken derivative.
	 */
	Content derivative0(Function f);
	
	
	/**
	 * Getting size of next content list.
	 * @return size of next content list.
	 */
	int getNextContentSize();
	
	
	/**
	 * Getting next content at specified index.
	 * @param index specified index.
	 * @return next content nextContent.
	 */
	Content getNextContent(int index);
	
	
	/**
	 * Finding index of next content.
	 * @param nextContent next content.
	 * @return index of next content.
	 */
	int indexOfNextContent(Content nextContent);
	
	
	/**
	 * Adding next content.
	 * @param nextContent next content.
	 * @return true if adding is successful.
	 */
	boolean addNextContent(Content nextContent);
	
	
	/**
	 * Removing next content at specified index.
	 * @param index specified index.
	 * @return removed next content.
	 */
	Content removeNextContent(int index);
	
	
	/**
	 * Removing next content.
	 * @param nextContent specified content.
	 * @return whether removing is successful.
	 */
	boolean removeNextContent(Content nextContent);

	
	/**
	 * Clearing next content list.
	 */
	void clearNextContents();
	
	
	/**
	 * Forwarding to evaluate the next content.
	 * @param nextContent next content.
	 * @return the filtered data.
	 */
	NeuronRaster forward(Content nextContent);

	
	/**
	 * Forwarding to evaluate the next content.
	 * @param nextContent next content.
	 * @param filter specified filter.
	 * @return the filtered data.
	 */
	NeuronRaster forward(Content nextContent, Filter filter);


}
