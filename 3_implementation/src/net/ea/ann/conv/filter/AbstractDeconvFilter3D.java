/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

/**
 * This class is an abstract implementation of deconvolution filter in 3D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class AbstractDeconvFilter3D extends AbstractFilter3D implements DeconvFilter3D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	protected AbstractDeconvFilter3D() {
		super();
	}


}
