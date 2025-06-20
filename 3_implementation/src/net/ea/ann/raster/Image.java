/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.nio.file.Path;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represent an image.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Image extends Serializable, Cloneable {

	
	/**
	 * Default source image type.
	 */
	int SOURCE_IMAGE_TYPE_DEFAULT = BufferedImage.TYPE_INT_ARGB;
	
	
	/**
	 * Name of default alpha value.
	 */
	String ALPHA_FIELD = "image_alpha";
	
	
	/**
	 * Default alpha value which is totally opaque. If it is zero, it is totally transparent.
	 */
	int ALPHA_DEFAULT = 255;


	/**
	 * Name of default image format.
	 */
	String IMAGE_FORMAT_DEFAULT = "png";


	/**
	 * Name of GIF image format.
	 */
	String IMAGE_FORMAT_GIF = "gif";


	/**
	 * Name of default animation format.
	 */
	String VIDEO_FORMAT_DEFAULT = "gif";


	/**
	 * Getting image width.
	 * @return image width.
	 */
	int getWidth();
	
	
	/**
	 * Getting image height.
	 * @return image height.
	 */
	int getHeight();


	/**
	 * Save image to path.
	 * @param path specified path.
	 * @return true if writing is successful.
	 */
	boolean save(Path path);
	

	/**
	 * Extracting image into neuron value array.
	 * @param neuronChannel neuron channel.
	 * @param width image width.
	 * @param height image height.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return neuron value array.
	 */
	NeuronValue[] convertFromImageToNeuronValues(int neuronChannel, int width, int height,
		boolean isNorm);

	
	/**
	 * Getting default format of image.
	 * @return default format of image.
	 */
	static String getDefaultFormat() {
		return IMAGE_FORMAT_DEFAULT;
	}

	
}
