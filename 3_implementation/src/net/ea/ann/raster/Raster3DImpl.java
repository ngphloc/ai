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
 * This class represents a raster in 3D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Raster3DImpl extends RasterAbstract implements Raster3D {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal image list.
	 */
	protected ImageList imageList = null;
	
	
	/**
	 * Constructor with image list.
	 * @param imageList image list.
	 */
	protected Raster3DImpl(ImageList imageList) {
		super();
		this.imageList = imageList;
	}


	@Override
	public int getWidth() {
		return imageList.getWidth();
	}


	@Override
	public int getHeight() {
		return imageList.getHeight();
	}


	@Override
	public int getDepth() {
		return imageList.size();
	}

	
	/**
	 * Setting depth.
	 * @param depth specified depth.
	 */
	private void setDepth(int depth) {
		imageList.setSize(depth);
	}
	
	
	@Override
	public java.awt.Image getRepImage() {
		if (imageList == null || imageList.size() == 0) return null;
		Image image = imageList.get(0);
		if (image != null && image instanceof ImageWrapper)
			return ((ImageWrapper)image).getImage();
		else
			return null;
	}


	@Override
	public String getDefaultFormat() {
		return ImageList.getDefaultFormat();
	}


	@Override
	public boolean save(Path path) {
		return imageList.save(path);
	}


	@Override
	public NeuronValue[] toNeuronValues(ConvLayerSingle layer, boolean isNorm) {
		NeuronValue[] values = imageList.convertFromImageToNeuronValues(layer.getNeuronChannel(), layer.getWidth(), layer.getHeight(),
				isNorm);
		int length = layer.getWidth() * layer.getHeight() * layer.getDepth(); 
		return NeuronValue.adjustArray(values, length, layer);
	}


	@Override
	public NeuronValue[] toNeuronValues(int neuronChannel, Size size, boolean isNorm) {
		if (size.depth != getDepth()) setDepth(size.depth);
		if (size.depth != getDepth()) {
			System.out.println("Different depth occurs in method Raster3DImpl.toNeuronValues(int, Size, boolean)");
			return null;
		}
		else
			return imageList.convertFromImageToNeuronValues(neuronChannel, size.width, size.height,
				isNorm);
	}
	
	
	/**
	 * Creating raster from image list.
	 * @param imageList specified image list.
	 * @return raster created from image list.
	 */
	public static Raster3DImpl create(ImageList imageList) {
		if (imageList != null && imageList.size() > 0)
			return new Raster3DImpl(imageList);
		else
			return null;
	}
	
	
	/**
	 * Create raster from neuron values.
	 * @param layer specified layer.
	 * @param values neuron values.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return raster created from neuron values.
	 */
	public static Raster3DImpl create(ConvLayerSingle layer, NeuronValue[] values,
			boolean isNorm, int defaultAlpha) {
		if (layer == null) return null;
		
		ImageList imageList = ImageList.convertFromNeuronValuesToImage(values, layer.getNeuronChannel(), layer.getWidth(), layer.getHeight(), layer.getDepth(), isNorm, defaultAlpha);
		return create(imageList);
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
	public static Raster3DImpl create(NeuronValue[] values, int neuronChannel, Size size,
			boolean isNorm, int defaultAlpha) {
		ImageList imageList = ImageList.convertFromNeuronValuesToImage(values, neuronChannel, size.width, size.height, size.depth, isNorm, defaultAlpha);
		return create(imageList);
	}

	
	/**
	 * Loading 3D raster from path.
	 * @param path specific path.
	 * @return 3D raster loaded from path.
	 */
	public static Raster3DImpl load(Path path) {
		ImageList imageList = ImageList.load(path);
		if (imageList != null)
			return new Raster3DImpl(imageList);
		else
			return null;
	}


}
