/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.Image;
import java.nio.file.Path;

import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class is a wrapper of raster.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RasterWrapper implements Raster {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal raster.
	 */
	protected Raster raster = null;
	
	
	/**
	 * Internal path if existing.
	 */
	protected Path path = null;
	
	
	/**
	 * Constructor with raster and path.
	 * @param raster specified raster.
	 * @param path specified path.
	 */
	public RasterWrapper(Raster raster, Path path) {
		this.raster = raster;
		this.path = path;
	}

	
	/**
	 * Constructor with raster.
	 * @param raster specified raster.
	 */
	public RasterWrapper(Raster raster) {
		this(raster, null);
	}

	
	@Override
	public int getWidth() {
		return raster.getWidth();
	}

	
	@Override
	public int getHeight() {
		return raster.getHeight();
	}

	
	@Override
	public int getDepth() {
		return raster.getDepth();
	}

	
	@Override
	public int getTime() {
		return raster.getTime();
	}

	
	@Override
	public Image getRepImage() {
		return raster.getRepImage();
	}


	@Override
	public String getDefaultFormat() {
		return raster.getDefaultFormat();
	}


	@Override
	public RasterProperty getProperty() {
		return raster.getProperty();
	}


	@Override
	public boolean save(Path path) {
		return raster.save(path);
	}

	
	@Override
	public NeuronValue[] toNeuronValues(ConvLayerSingle layer, boolean isNorm) {
		return raster.toNeuronValues(layer, isNorm);
	}

	
	@Override
	public NeuronValue[] toNeuronValues(int neuronChannel, Size size, boolean isNorm) {
		return raster.toNeuronValues(neuronChannel, size, isNorm);
	}

	
	/**
	 * Getting path.
	 * @return internal path.
	 */
	public Path getPath() {
		if (path != null)
			return path;
		else
			return raster instanceof RasterWrapper ? ((RasterWrapper)raster).getPath() : null;
	}
	
	
	/**
	 * Setting path.
	 * @param path specified path.
	 */
	public void setPath(Path path) {
		this.path = path;
	}
	
	
	/**
	 * Getting name of this raster from path.
	 * @return name of this raster from path.
	 */
	public String getName() {
		if (path == null) return null;
		Path fileName = path.getFileName();
		return fileName != null ? fileName.toString() : null;
	}
	
	
	/**
	 * Getting plain name of this raster from path.
	 * @return plain name of this raster from path.
	 */
	public String getNamePlain() {
		String name = getName();
		if (name == null)
			return null;
		else if (name.isEmpty() || name.equals("."))
			return "";
		else
			return name.contains(".") ? name.substring(0, name.lastIndexOf(".")) : name;
	}
	
	
}
