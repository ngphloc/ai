/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import javax.imageio.ImageIO;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.raster.Raster.RasterType;

/**
 * This class is a serializable wrapper of Java image.
 * Please review the serialization technique available at <a href = "https://stackoverflow.com/questions/15058663/how-to-serialize-an-object-that-includes-bufferedimages">https://stackoverflow.com/questions/15058663/how-to-serialize-an-object-that-includes-bufferedimages</a>
 * because there are some unexpected problems when RMI cannot serialize this BufferedImage with given {@link #readObject(ObjectInputStream)} and {@link #writeObject(ObjectOutputStream)}.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ImageWrapper implements Image {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal transient image. Please review the serialization technique available at <a href = "https://stackoverflow.com/questions/15058663/how-to-serialize-an-object-that-includes-bufferedimages">https://stackoverflow.com/questions/15058663/how-to-serialize-an-object-that-includes-bufferedimages</a>
	 * because there are some unexpected problems when RMI cannot serialize this BufferedImage with given {@link #readObject(ObjectInputStream)} and {@link #writeObject(ObjectOutputStream)}.
	 */
	protected transient BufferedImage image;
	
	
	/**
	 * The auxiliary image data supports serialization. Constructor {@link #ImageWrapper(BufferedImage)} will create such data from the buffered image.
	 * Method {@link #getImage()} will create the buffered image from such data if the the buffered image is null by non-serializing.
	 * The value imageData[0] is image width and the value imageData[1] is image height. Image pixels are stored from imageData[2]. 
	 */
	protected int[] imageData = null;
	
	
	/**
	 * Constructor with image. 
	 * @param image specific image.
	 */
	public ImageWrapper(BufferedImage image) {
		this.image = image;
		
		try {
			//The value imageData[0] is image width and the value imageData[1] is image height. Image pixels are stored from imageData[2].
			if (this.image != null && this.imageData == null)
				this.imageData = convertFromImageToData(this.image);
		} catch (Throwable e) {
			this.imageData = null;
			System.out.println("Error: " + e.getMessage());
		}
	}

	
	/**
	 * Getting internal image. This method will be improved to support serialization in the next version.
	 * @return internal image.
	 */
	protected BufferedImage getImage() {
		try {
			//The value imageData[0] is image width and the value imageData[1] is image height. Image pixels are stored from imageData[2].
			if (this.image == null && this.imageData != null)
				this.image = convertFromDataToImage(imageData);
		} catch (Throwable e) {
			System.out.println("Error: " + e.getMessage());
		}
		
		return this.image;
	}
	

	@Override
	public int getWidth() {
		return getImage().getWidth();
	}
	

	@Override
	public int getHeight() {
		return getImage().getHeight();
	}
	

	/**
	 * Writing object for serialization.
	 * @param out specific output stream.
	 * @throws IOException if IO errors raise.
	 */
	private void writeObject(ObjectOutputStream out) throws IOException {
		out.defaultWriteObject();
		ImageIO.write(getImage(), Image.IMAGE_FORMAT_DEFAULT, out);
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
    
    
	/*
	 * The code is available at https://stackoverflow.com/questions/3514158/how-do-you-clone-a-bufferedimage
	 */
	@Override
	protected Object clone() throws CloneNotSupportedException {
		BufferedImage image = getImage();
		if (image == null)
			return super.clone();
		else {
			ColorModel cm = image.getColorModel();
			boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
			WritableRaster raster = image.copyData(null);
			BufferedImage bi = new BufferedImage(cm, raster, isAlphaPremultiplied, null);
			return new ImageWrapper(bi);
		}
	}


	/**
	 * Save image to path.
	 * @param path image path.
	 * @param imageFormat image format.
	 * @return true if writing is successful.
	 */
	private boolean save(Path path, String imageFormat) {
		try {
			BufferedImage image = getImage();
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


	@Override
	public boolean save(Path path) {
		return save(path, Image.IMAGE_FORMAT_DEFAULT);
	}

		
	/**
	 * Loading image from path.
	 * @param path specific path.
	 * @return image loaded from path.
	 */
	public static ImageWrapper load(Path path) {
		try {
			InputStream is = Files.newInputStream(path);
			BufferedImage image = ImageIO.read(is);
			is.close();
			
			if (image == null)
				return null;
			else
				return new ImageWrapper(image);
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;

	}
	
	
	/**
	 * Converting raster type to neuron channel. 
	 * @param RasterType raster type.
	 * @return neuron channel
	 */
	private static int toImageType(RasterType rasterType) {
		int imageType = 1;
        switch (rasterType) {
        case GRAY:
        	imageType = Image.SOURCE_IMAGE_TYPE_DEFAULT;
        	break;
        case GB:
        	imageType = Image.SOURCE_IMAGE_TYPE_DEFAULT;
        	break;
        case RGB:
        	imageType = Image.SOURCE_IMAGE_TYPE_DEFAULT;
        	break;
        case ARGB:
        	imageType = Image.SOURCE_IMAGE_TYPE_DEFAULT;
        	break;
        default:
        	imageType = Image.SOURCE_IMAGE_TYPE_DEFAULT;
        	break;
        }
        
        return imageType;
	}
	
	
	/**
	 * Resize image.
	 * @param image specific image.
	 * @param newWidth new width.
	 * @param newHeight new height.
	 * @return resized image.
	 */
	protected static BufferedImage resize(BufferedImage image, int newWidth, int newHeight) {
		if (image == null || newWidth <= 0 || newHeight <= 0)
			return null;
		else if (image.getWidth() != newWidth || image.getHeight() != newHeight) {
			int sourceImageType = image.getType();
			java.awt.Image resizedImage = image.getScaledInstance(newWidth, newHeight, java.awt.Image.SCALE_DEFAULT);
			if (resizedImage == null) return null;
			
			image = convertToSourceTypeImage(resizedImage, sourceImageType);
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
	 * @param sourceImageType source image type. See {@link BufferedImage#getType()}.
	 * @return buffered image.
	 */
	private static BufferedImage convertToSourceTypeImage(java.awt.Image image, int sourceImageType) {
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
	 * @param rasterType raster type.
	 * @param width image width.
	 * @param height image height.
	 * @param sourceImageType source image type.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return converted image.
	 */
	private static BufferedImage convertFromNeuronValuesToImage(NeuronValue[] values, RasterType rasterType, int width, int height,
			int sourceImageType, boolean isNorm, int defaultAlpha) {
		if (values == null || values.length == 0 || width <= 0 || height <= 0) return null;
		
		BufferedImage image = new BufferedImage(width, height, sourceImageType);
		
		double factor = isNorm ? 255 : 1;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int a = defaultAlpha, r = 0, g = 0, b = 0, gray = 0;
				
				int index = y*width + x;
				if (index < values.length) {
	                switch (rasterType) {
	                case GRAY:
						NeuronValue1 value1 = (NeuronValue1)(values[index]);
						gray = (int)(value1.get()*factor + 0.5);
						r = g = b = gray;
	                	break;
	                case GB:
						NeuronValueV value2 = (NeuronValueV)(values[index]);
	                	g = (int)(value2.get(0)*factor + 0.5);
	                	b = (int)(value2.get(1)*factor + 0.5);
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
	 * Converting neuron values to image.
	 * @param values neuron values.
	 * @param neuronChannel neuron channel.
	 * @param width image width.
	 * @param height image height.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return image converted from neuron values.
	 */
	public static ImageWrapper convertFromNeuronValuesToImage(NeuronValue[] values, int neuronChannel, int width, int height,
			boolean isNorm, int defaultAlpha) {
		RasterType rasterType = Raster.toRasterType(neuronChannel);
		BufferedImage image = convertFromNeuronValuesToImage(values, rasterType, width, height,
			toImageType(rasterType), isNorm, defaultAlpha);
		return image != null ? new ImageWrapper(image) : null;
	}
	

	/**
	 * Converting data to image.
	 * @param imageData specified image data.
	 * The value imageData[0] is image width and the value imageData[1] is image height. Image pixels are stored from imageData[2].
	 * @return image.
	 */
	private static BufferedImage convertFromDataToImage(int[] imageData) {
		if (imageData == null || imageData.length <= 2) return null;
		
		int width = imageData[0];
		int height = imageData[1];
		if (width <= 0 || height <= 0) return null;
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int p = imageData[2 + y*width + x];
	            image.setRGB(x, y, p);
			}
		}
		
		return image;
	}
	
	
	/**
	 * Extracting image into neuron value array.
	 * @param rasterType raster type.
	 * @param width image width.
	 * @param height image height.
	 * @param image specific Java image.
	 * @param sourceImageType source image type.
	 * @param isResize flag to indicate whether image is resized.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return neuron value array.
	 */
	private static NeuronValue[] convertFromImageToNeuronValues(RasterType rasterType, int width, int height,
			BufferedImage image, int sourceImageType, boolean isResize, boolean isNorm) {
		if (image == null || width <= 0 || height <= 0) return null;
		
		if (isResize && image.getWidth() != width && image.getHeight() != height) {
			image = resize(image, width, height);
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
	            	value = createNeuronValue(rasterType, 0, 0, 0, 0, 0, 1);
	            else {
					int p = image.getRGB(x, y);
					  
		            int a = (p >> 24) & 0xff;
		            int r = (p >> 16) & 0xff;
		            int g = (p >> 8) & 0xff;
		            int b = p & 0xff;
		            
		            //Gray value
		            int gray = (r + g + b) / 3;
		            
	            	value = createNeuronValue(rasterType, a, r, g, b, gray, factor);
	            }
	
	            values[y*width + x] = value;
			}
			
		}
		
		return values;
	}


	/**
	 * Create neuron value.
	 * @param rasterType raster type.
	 * @param a alpha value.
	 * @param r red value.
	 * @param g green value.
	 * @param b blue value.
	 * @param gray gray value.
	 * @param factor specific factor.
	 * @return neuron value.
	 */
	private static NeuronValue createNeuronValue(RasterType rasterType, int a, int r, int g, int b, int gray, double factor) {
        NeuronValue value = null;
        switch (rasterType) {
        case GRAY:
        	value = new NeuronValue1((double)gray/factor);
        	break;
        case GB:
        	value = new NeuronValueV((double)g/factor, (double)b/factor);
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
	 * Converting image to data.
	 * @param image specified image.
	 * @return image data.
	 * The value imageData[0] is image width and the value imageData[1] is image height. Image pixels are stored from imageData[2].
	 */
	private static int[] convertFromImageToData(BufferedImage image) {
		if (image == null) return null;
		int width = image.getWidth(), height = image.getHeight();
		if (width == 0 || height == 0) return null;
		
		int[] imageData = new int[2 + width*height];
		imageData[0] = width;
		imageData[1] = height;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int p = image.getRGB(x, y);
				imageData[2 + y*width + x] = p;
			}
		}
		
		return imageData;
	}
	
	
	@Override
	public NeuronValue[] convertFromImageToNeuronValues(int neuronChannel, int width, int height,
			boolean isNorm) {
		RasterType rasterType = Raster.toRasterType(neuronChannel);
		return convertFromImageToNeuronValues(rasterType, width, height,
			getImage(), toImageType(rasterType), true, isNorm);
	}
	
	
}
