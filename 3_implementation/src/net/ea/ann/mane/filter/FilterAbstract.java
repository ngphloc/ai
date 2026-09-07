/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.mane.Filter;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.mane.Parameter;

/**
 * This class is an abstract implementation of filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class FilterAbstract implements Filter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Flag to indicate whether to move according to stride when filtering.
	 */
	protected boolean moveStride = MOVE_STRIDE;

	
	/**
	 * Default constructor.
	 */
	protected FilterAbstract() {
		super();
	}

	
	@Override
	public boolean isMoveStride() {return moveStride;}


	@Override
	public void setMoveStride(boolean moveStride) {this.moveStride = moveStride;}


	@Override
	public Parameter pcopy(Parameter other) {
		Filter.super.pcopy(other);
		
		this.moveStride = ((FilterAbstract)other).moveStride;
		return this;
	}


	/**
	 * Getting network.
	 * @return network;
	 */
	MatrixNetworkAbstract getNetwork() {return getLayer() != null ? getLayer().getNetwork() : null;}
	
	
	/**
	 * Checking whether to make gradient clipping.
	 * @return whether to make gradient clipping.
	 */
	boolean isGradClipping() {return getNetwork() != null ? getNetwork().paramIsGradClipping() : false;}
	
	
	/**
	 * Getting maximum gradient norm for gradient clipping.
	 * The value ranges from 1.0 to 5.0. The value 0 indicates no gradient clipping.
	 * @return raster channel.
	 */
	double getGradNormMax() {return getNetwork() != null ? getNetwork().paramGetGradNormMax() : 0;}

	

}

