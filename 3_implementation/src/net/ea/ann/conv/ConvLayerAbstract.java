/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerAbstract;
import net.ea.ann.core.NeuronValue;

/**
 * This class is an abstract implementation of convolutional layer.
 * 
 * @author Loc Nguyen
 *
 */
public abstract class ConvLayerAbstract extends LayerAbstract implements ConvLayer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Internal data as array of neurons.
	 */
	protected ConvNeuron[][] data = null;
	

	/**
	 * Constructor with ID reference.
	 * @param idRef ID reference.
	 */
	protected ConvLayerAbstract(Id idRef) {
		super(idRef);
	}


	/**
	 * Default constructor.
	 */
	public ConvLayerAbstract() {
		this(null);
	}


	@Override
	public ConvNeuron[][] forward() {
		ConvLayer nextLayer = getNextLayer();
		if (nextLayer == null) return null;
		ConvNeuron[][] data = nextLayer.getData();
		if (data == null || data.length == 0) return null;
		
		Filter filter = getFilter();
		int filterWidth = filter.width();
		int filterHeight = filter.height();
		int widthBlock = this.getWidth() / filterWidth;
		int heightBlock = this.getHeight() / filterHeight;
		
		int width = nextLayer.getWidth();
		int height = nextLayer.getHeight();
		for (int y = 0; y < height; y++) {
			int yBlock = y < heightBlock ? y : heightBlock-1;
			int Y = yBlock*filterHeight;
			for (int x = 0; x < width; x++) {
				int xBlock = x < widthBlock ? x : widthBlock-1;
				int X = xBlock*filterWidth;
				
				NeuronValue value = filter.apply(X, Y, this);
				if (value != null)
					data[y][x].setValue(value);
				else
					data[y][x].setValue(newNeuronValue().zero());
			}
		}
		
		return data;
	}

	
}
