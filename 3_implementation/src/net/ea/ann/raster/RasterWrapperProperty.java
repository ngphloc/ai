/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.nio.file.Path;

/**
 * This class is a wrapper of raster and property.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RasterWrapperProperty extends RasterWrapper {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal property.
	 */
	protected RasterProperty property = null;
	
	
	/**
	 * Constructor with raster and path.
	 * @param raster specified raster.
	 * @param path specified path.
	 */
	public RasterWrapperProperty(Raster raster, Path path) {
		super(raster, path);
		if (raster != null) this.property = raster.getProperty().shallowDuplicate();
	}

	
	/**
	 * Constructor with raster.
	 * @param raster specified raster.
	 */
	public RasterWrapperProperty(Raster raster) {
		this(raster, null);
	}


	@Override
	public RasterProperty getProperty() {
		return property != null ? property : super.getProperty();
	}

	
	/**
	 * Setting property.
	 * @param property specified property.
	 */
	public void setProperty(RasterProperty property) {
		this.property = property;
	}
	
	
}
