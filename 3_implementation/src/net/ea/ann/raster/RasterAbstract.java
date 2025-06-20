/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

/**
 * This class is an abstract implementation of raster.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class RasterAbstract implements Raster {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal raster property.
	 */
	protected RasterProperty property = new RasterPropertyImpl();
	
	
	/**
	 * Default constructor.
	 */
	protected RasterAbstract() {
		super();
	}


	@Override
	public int getHeight() {
		return 1;
	}


	@Override
	public int getDepth() {
		return 1;
	}


	@Override
	public int getTime() {
		return 1;
	}
	
	
	@Override
	public String getDefaultFormat() {
		return Image.getDefaultFormat();
	}


	@Override
	public RasterProperty getProperty() {
		return property;
	}

	
}
