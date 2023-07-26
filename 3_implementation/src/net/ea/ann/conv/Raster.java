/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import javax.imageio.ImageIO;

import net.ea.ann.core.NeuronValue;
import net.ea.ann.core.NeuronValue1;
import net.ea.ann.core.NeuronValueV;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.LogisticFunction1;
import net.ea.ann.core.function.LogisticFunctionV;

/**
 * This class represent an serializable raster.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Raster implements Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of default source image type.
	 */
	public static final String SOURCE_IMAGE_TYPE_FIELD = "conv_source_image_type";
	

	/**
	 * Default source image type.
	 */
	public static final int SOURCE_IMAGE_TYPE_DEFAULT = BufferedImage.TYPE_INT_ARGB;
	
	
	/**
	 * Name of flag to normalize pixel in rang [0, 1].
	 */
	public static final String NORM_FIELD = "conv_norm";

	
	/**
	 * Flag to normalize pixel in rang [0, 1].
	 */
	public static final boolean NORM_DEFAULT = true;

	
	/**
	 * Name of resizing source image flag.
	 */
	public static final String SOURCE_RESIZE_FIELD = "conv_source_resize";

	
	/**
	 * Resizing source image flag.
	 */
	public static final boolean SOURCE_RESIZE_DEFAULT = true;
	
	
	/**
	 * Name of default alpha value.
	 */
	public static final String ALPHA_FIELD = "conv_alpha";

	
	/**
	 * Default alpha value which is totally opaque.
	 */
	public static final int ALPHA_DEFAULT = 255;
	
	
	/**
	 * This enum represents image type for this convolutional VAE.
	 * @author Loc Nguyen
	 *
	 */
	public enum ImageType {
		
		/**
		 * Gray single channel.
		 */
		GRAY,
		
		/**
		 * RGB triple channel
		 */
		RGB,
		
		/**
		 * ARGB quadruplet channel
		 */
		ARGB,
		
	}
	

	/**
	 * Name of default source image format.
	 */
	public static final String IMAGE_FORMAT_DEFAULT = "png";

	
	/**
	 * Internal transient image.
	 */
	protected transient BufferedImage image;
	
	
	/**
	 * Constructor with image. 
	 * @param image specific image.
	 */
	public Raster(BufferedImage image) {
		this.image = image;
	}

	
	/**
	 * Getting internal image.
	 * @return internal image.
	 */
	public BufferedImage getImage() {
		return image;
	}
	
	
	/**
	 * Getting raster width.
	 * @return raster width.
	 */
	public int getWidth() {
		return image.getWidth();
	}
	
	
	/**
	 * Getting raster height.
	 * @return raster height.
	 */
	public int getHeight() {
		return image.getHeight();
	}
	

	/**
	 * Writing object for serialization.
	 * @param out specific output stream.
	 * @throws IOException if IO errors raise.
	 */
	private void writeObject(ObjectOutputStream out) throws IOException {
		out.defaultWriteObject();
		ImageIO.write(image, IMAGE_FORMAT_DEFAULT, out);
	}

	
	/**
	 * Reading object for serialization.
	 * @param in input stream.
	 * @throws IOException if IO errors raise.
	 * @throws ClassNotFoundException if no class is found.
	 */
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
		in.defaultReadObject();
		ImageIO.read(in);
	}
    
    
	/**
	 * Save raster to path.
	 * @param path raster path.
	 * @param imageFormat image format.
	 * @return true if writing is successful.
	 */
	private boolean save(Path path, String imageFormat) {
		try {
			if (image == null) return false;
			
			OutputStream os = Files.newOutputStream(path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
			ImageIO.write(image, imageFormat, os);
			os.close();
			
			return true;
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return false;
	}


	/**
	 * Save raster to path.
	 * @param path raster path.
	 * @return true if writing is successful.
	 */
	public boolean save(Path path) {
		return save(path, IMAGE_FORMAT_DEFAULT);
	}

		
	/**
	 * Creating this raster from path.
	 * @param path specific path.
	 * @return raster loaded from path.
	 */
	public static Raster load(Path path) {
		try {
			InputStream is = Files.newInputStream(path);
			BufferedImage image = ImageIO.read(is);
			is.close();
			
			if (image == null)
				return null;
			else
				return new Raster(image);
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;

	}
	
	
//	/**
//	 * Converting image type to neuron channel. 
//	 * @param imageType image type.
//	 * @return neuron channel
//	 */
//	private static int toNeuronChannel(ImageType imageType) {
//		int channel = 1;
//        switch (imageType) {
//        case GRAY:
//        	channel = 1;
//        	break;
//        case RGB:
//        	channel = 3;
//        	break;
//        case ARGB:
//        	channel = 4;
//        	break;
//        default:
//        	channel = 1;
//        	break;
//        }
//        
//        return channel;
//	}
	
	
	/**
	 * Convert neuron channel to image type.
	 * @param neuronChannel neuron channel.
	 * @return image type.
	 */
	private static ImageType toImageType(int neuronChannel) {
		ImageType imageType = ImageType.GRAY;
        switch (neuronChannel) {
        case 1:
        	imageType = ImageType.GRAY;
        	break;
        case 3:
        	imageType = ImageType.RGB;
        	break;
        case 4:
        	imageType = ImageType.ARGB;
        	break;
        default:
        	imageType = ImageType.GRAY;
        	break;
        }
        
        return imageType;
	}
	
	
	/**
	 * Converting image type to neuron channel. 
	 * @param imageType image type.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return neuron channel
	 */
	private static Function toActivationRef(ImageType imageType, boolean isNorm) {
		Function f = null;
		
        switch (imageType) {
        case GRAY:
        	f = isNorm ? new LogisticFunction1(0.0, 1.0) : new LogisticFunction1(0.0, 255.0);
        	break;
        case RGB:
        	f = isNorm ? new LogisticFunctionV(3, 0.0, 1.0) : new LogisticFunctionV(3, 0.0, 255.0);
        	break;
        case ARGB:
        	f = isNorm ? new LogisticFunctionV(4, 0.0, 1.0) : new LogisticFunctionV(4, 0.0, 255.0);
        	break;
        default:
        	f = isNorm ? new LogisticFunction1(0.0, 1.0) : new LogisticFunction1(0.0, 255.0);
        	break;
        }
        
        return f;
	}


	/**
	 * Converting image type to neuron channel. 
	 * @param neuronChannel neuron channel.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return neuron channel
	 */
	public static Function toActivationRef(int neuronChannel, boolean isNorm) {
		if (isNorm) {
			if (neuronChannel <= 0)
				return null;
			else if (neuronChannel == 1)
	        	return new LogisticFunction1(0.0, 1.0);
			else
	        	return new LogisticFunctionV(neuronChannel, 0.0, 1.0);
		}
		
		ImageType imageType = toImageType(neuronChannel);
		return toActivationRef(imageType, isNorm);
	}

	
	/**
	 * Resize image.
	 * @param image specific image.
	 * @param newWidth new width.
	 * @param newHeight new height.
	 * @param sourceImageType source image type.
	 * @return resized image.
	 */
	private static BufferedImage resize(BufferedImage image, int newWidth, int newHeight, int sourceImageType) {
		if (image == null || newWidth <= 0 || newHeight <= 0)
			return null;
		else if (image.getWidth() != newWidth || image.getHeight() != newHeight) {
			Image resizedImage = image.getScaledInstance(newWidth, newHeight, Image.SCALE_DEFAULT);
			if (resizedImage == null) return null;
			
			image = Raster.convertToSourceTypeImage(resizedImage, sourceImageType);
			if (image == null)
				return null;
			//else if (image.getWidth() != newWidth || image.getHeight() != newHeight)
			//	return null;
			else
				return image;
		}
		else
			return image;
	}


	/**
	 * Converting image to source type image. The code is available at https://stackoverflow.com/questions/13605248/java-converting-image-to-bufferedimage 
	 * @param image specific image.
	 * @param sourceImageType source image type.
	 * @return buffered image.
	 */
	private static BufferedImage convertToSourceTypeImage(Image image, int sourceImageType) {
		if (image == null) return null;
		
		if (image instanceof BufferedImage) {
			BufferedImage bufferedImage = (BufferedImage)image;
			if (bufferedImage.getType() == sourceImageType) return bufferedImage;
		}
	
		BufferedImage bufferedImage = new BufferedImage(image.getWidth(null), image.getHeight(null), sourceImageType);
	    Graphics2D g = bufferedImage.createGraphics();
	    g.drawImage(image, 0, 0, null);
	    g.dispose();
	
	    return bufferedImage;
	}


	/**
	 * Converting neuron values to image.
	 * @param values neuron values.
	 * @param imageType image type.
	 * @param width raster width.
	 * @param height raster height.
	 * @param sourceImageType source image type.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return converted image.
	 */
	private static BufferedImage convertFromNeuronValuesToImage(NeuronValue[] values, ImageType imageType, int width, int height,
			int sourceImageType, boolean isNorm, int defaultAlpha) {
		if (values == null || values.length == 0 || width <= 0 || height <= 0) return null;
		
		BufferedImage image = new BufferedImage(width, height, sourceImageType);
		
		double factor = isNorm ? 255 : 1;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int a = defaultAlpha, r = 0, g = 0, b = 0, gray = 0;
				
				int index = y*width + x;
				if (index < values.length) {
	                switch (imageType) {
	                case GRAY:
						NeuronValue1 value1 = (NeuronValue1)(values[index]);
						gray = (int)(value1.get()*factor + 0.5);
						r = g = b = gray;
	                	break;
	                case RGB:
						NeuronValueV value3 = (NeuronValueV)(values[index]);
	                	r = (int)(value3.get(0)*factor + 0.5);
	                	g = (int)(value3.get(1)*factor + 0.5);
	                	b = (int)(value3.get(2)*factor + 0.5);
	                	break;
	                case ARGB:
						NeuronValueV value4 = (NeuronValueV)(values[index]);
	                	a = (int)(value4.get(0)*factor + 0.5);
	                	r = (int)(value4.get(1)*factor + 0.5);
	                	g = (int)(value4.get(2)*factor + 0.5);
	                	b = (int)(value4.get(3)*factor + 0.5);
	                	break;
	                default:
						NeuronValue1 d = (NeuronValue1)(values[index]);
						gray = (int)(d.get()*factor + 0.5);
						r = g = b = gray;
	                	break;
	                }
				}
				
				int p = (a << 24) | (r << 16) | (g << 8) | b;
	            image.setRGB(x, y, p);
	            
			} //End for x
			
		} //End for y
		
		return image;
	}


	/**
	 * Converting neuron values to raster.
	 * @param values neuron values.
	 * @param neuronChannel neuron channel.
	 * @param width raster width.
	 * @param height raster height.
	 * @param sourceImageType source image type.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return converted raster specification.
	 */
	public static Raster convertFromNeuronValuesToRaster(NeuronValue[] values, int neuronChannel, int width, int height,
			int sourceImageType, boolean isNorm, int defaultAlpha) {
		BufferedImage image = convertFromNeuronValuesToImage(values, toImageType(neuronChannel), width, height,
				sourceImageType, isNorm, defaultAlpha);
		return image != null ? new Raster(image) : null;
	}
	
	
	/**
	 * Extracting image into neuron value array.
	 * @param imageType image type.
	 * @param width raster width.
	 * @param height raster height.
	 * @param image specific image.
	 * @param sourceImageType source image type.
	 * @param isResize flag to indicate whether image is resized.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return neuron value array.
	 */
	private static NeuronValue[] convertFromImageToNeuronValues(ImageType imageType, int width, int height,
			BufferedImage image, int sourceImageType, boolean isResize, boolean isNorm) {
		if (image == null || width <= 0 || height <= 0) return null;
		
		if (isResize && image.getWidth() != width && image.getHeight() != height) {
			image = resize(image, width, height, sourceImageType);
			if (image == null) return null;
		}
		
		if (image.getType() != sourceImageType) {
			image = convertToSourceTypeImage(image, sourceImageType);
			if (image == null) return null;
		}
		
		if (image.getWidth() <= 0 && image.getHeight() <= 0) return null;
	
		NeuronValue[] values = new NeuronValue[width*height];
	
		double factor = isNorm ? 255 : 1;
		int minWidth = Math.min(width, image.getWidth());
		int minHeight = Math.min(height, image.getHeight());
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
	            NeuronValue value = null;
	            if (x >= minWidth || y >= minHeight)
	            	value = createNeuronValue(imageType, 0, 0, 0, 0, 0, 1);
	            else {
					int p = image.getRGB(x, y);
					  
		            int a = (p >> 24) & 0xff;
		            int r = (p >> 16) & 0xff;
		            int g = (p >> 8) & 0xff;
		            int b = p & 0xff;
		            
		            //Gray value
		            int gray = (r + g + b) / 3;
		            
	            	value = createNeuronValue(imageType, a, r, g, b, gray, factor);
	            }
	
	            values[y*width + x] = value;
			}
			
		}
		
		return values;
	}


	/**
	 * Create neuron value.
	 * @param imageType image type.
	 * @param a alpha value.
	 * @param r red value.
	 * @param g green value.
	 * @param b blue value.
	 * @param gray gray value.
	 * @param factor specific factor.
	 * @return neuron value.
	 */
	private static NeuronValue createNeuronValue(ImageType imageType, int a, int r, int g, int b, int gray, double factor) {
        NeuronValue value = null;
        switch (imageType) {
        case GRAY:
        	value = new NeuronValue1((double)gray/factor);
        	break;
        case RGB:
        	value = new NeuronValueV((double)r/factor, (double)g/factor, (double)b/factor);
        	break;
        case ARGB:
        	value = new NeuronValueV((double)a/factor, (double)r/factor, (double)g/factor, (double)b/factor);
        	break;
        default:
        	value = new NeuronValue1((double)gray/factor);
        	break;
        }
        
        return value;
	}
	
	
	/**
	 * Extracting raster into neuron value array.
	 * @param neuronChannel neuron channel.
	 * @param imageWidth image width.
	 * @param imageHeight image height.
	 * @param sourceImageType source image type.
	 * @param isResize flag to indicate whether image is resized.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return neuron value array.
	 */
	public NeuronValue[] convertFromRasterToNeuronValues(int neuronChannel, int imageWidth, int imageHeight,
			int sourceImageType, boolean isResize, boolean isNorm) {
		return convertFromImageToNeuronValues(toImageType(neuronChannel), imageWidth, imageHeight,
				this.getImage(), sourceImageType, isResize, isNorm);
	}
	
	
}
