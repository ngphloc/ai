/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

/**
 * This class represent an raster in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Raster2D extends Raster {

	
	/**
	 * Field of learning raster mode.
	 * In this mode (true), unified content will be learned back original content in convolutional neural network.
	 */
	String LEARN_FIELD = "raster_learn";
	
	
	/**
	 * Default value of learning raster mode.
	 * In this mode (true), unified content will be learned back original content in convolutional neural network.
	 */
	boolean LEARN_DEFAULT = false;
	
	
}
