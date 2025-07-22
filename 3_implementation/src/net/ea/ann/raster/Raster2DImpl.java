/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.nio.file.Path;

import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class is the default implementation of raster.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Raster2DImpl extends RasterAbstract implements Raster2D {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal image.
	 */
	protected Image image = null;
	
	
	/**
	 * Constructor with image wrapper.
	 * @param image Java image.
	 */
	protected Raster2DImpl(Image image) {
		super();
		this.image = image;
	}


	@Override
	public int getWidth() {
		return image.getWidth();
	}
	
	
	@Override
	public int getHeight() {
		return image.getHeight();
	}
	

	@Override
	public Image getImage() {return image;}
	
	
	@Override
	public java.awt.Image getRepImage() {
		if (image == null)
			return null;
		else if (image instanceof ImageWrapper)
			return ((ImageWrapper)image).image;
		else
			return null;
	}

	
	@Override
	public String getDefaultFormat() {
		return Image.getDefaultFormat();
	}


	@Override
	public boolean save(Path path) {
		return image.save(path);
	}

	
	@Override
	public NeuronValue[] toNeuronValues(ConvLayerSingle layer, boolean isNorm) {
		if (this.image == null || layer == null) return null;
		return this.image.convertFromImageToNeuronValues(layer.getNeuronChannel(), layer.getWidth(), layer.getHeight(),
			isNorm);
	}


	@Override
	public NeuronValue[] toNeuronValues(int neuronChannel, Size size,
			boolean isNorm) {
		return image.convertFromImageToNeuronValues(neuronChannel, size.width, size.height,
			isNorm);
	}

	
	/**
	 * Creating raster from specified image.
	 * @param image specified image.
	 * @return raster created from specified image.
	 */
	public static Raster2DImpl create(Image image) {
		if (image == null)
			return null;
		else
			return new Raster2DImpl(image);
	}
	
	
	/**
	 * Create raster from neuron values.
	 * @param layer specified layer.
	 * @param values neuron values.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return raster created from neuron values.
	 */
	public static Raster2DImpl create(ConvLayerSingle layer, NeuronValue[] values,
			boolean isNorm, int defaultAlpha) {
		if (layer == null) return null;
		
		ImageWrapper image = ImageWrapper.convertFromNeuronValuesToImage(values, layer.getNeuronChannel(), layer.getWidth(), layer.getHeight(),
			isNorm, defaultAlpha);
		return create(image);
	}
	
	
	/**
	 * Create raster from neuron values.
	 * @param values neuron values.
	 * @param neuronChannel neuron channel.
	 * @param size raster size.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return raster created from neuron values.
	 */
	public static Raster2DImpl create(NeuronValue[] values, int neuronChannel, Size size,
			boolean isNorm, int defaultAlpha) {
		ImageWrapper image = ImageWrapper.convertFromNeuronValuesToImage(values, neuronChannel, size.width, size.height,
			isNorm, defaultAlpha);
		return create(image);
	}


	/**
	 * Loading raster from path.
	 * @param path specific path.
	 * @return raster loaded from path.
	 */
	public static Raster2DImpl load(Path path) {
		ImageWrapper image = ImageWrapper.load(path);
		return create(image);
	}
	
	
}
