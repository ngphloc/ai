/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.raster.Size;

/**
 * This class represents a pooling filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class PoolFilter extends FilterAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Kernel width.
	 */
	protected int width = 1;
	
	
	/**
	 * Kernel height.
	 */
	protected int height = 1;

	
	/**
	 * Constructor with size.
	 * @param size size.
	 */
	protected PoolFilter(Size size) {
		super();
		if (size.width < 1 || size.height < 1) throw new IllegalArgumentException();
		this.width = size.width;
		this.height = size.height;
	}

	
	@Override
	public int width() {
		return width;
	}


	@Override
	public int height() {
		return height;
	}
	
	
	/**
	 * Forwarding evaluation from previous layers to current layers.
	 * @param prevLayer current layer.
	 * @param thisInputLayer current input layer.
	 * @param thisOutputLayer current output layer.
	 */
	public abstract void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer);

	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 */
	public abstract Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer);

	
}
