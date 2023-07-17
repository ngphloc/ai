/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.rmi.RemoteException;
import java.util.List;

import javax.imageio.ImageIO;

import net.ea.ann.Id;
import net.ea.ann.NeuronValue;
import net.ea.ann.NeuronValue1;
import net.ea.ann.NeuronValueV;
import net.ea.ann.Record;
import net.ea.ann.Util;
import net.ea.ann.function.Function;
import net.ea.ann.function.LogisticFunction1;
import net.ea.ann.function.LogisticFunctionV;

/**
 * This class represent an default implementation of convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvVAEImpl extends VAEImpl implements ConvVAE {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default source image type.
	 */
	private static final int DEFAULT_SOURCE_IMAGE_TYPE = BufferedImage.TYPE_INT_ARGB;
	

	/**
	 * Resizing source image flag.
	 */
	private static final boolean SOURCE_RESIZING = true;
	
	
	/**
	 * Default alpha value which is totally opaque.
	 */
	private static final int DEFAULT_ALPHA = 255;
	
	
	/**
	 * Flag to normalize pixel in rang [0, 1].
	 */
	protected static final boolean IS_NORM = true;

	
	/**
	 * This enum represents image type for this convolutional VAE.
	 * @author Loc Nguyen
	 *
	 */
	protected enum ImageType {
		
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
	 * Image type.
	 */
	protected ImageType imageType = ImageType.GRAY;
	
	
	/**
	 * Image width.
	 */
	protected int imageWidth = 0;
	
	
	/**
	 * Image width.
	 */
	protected int imageHeight = 0;

	
	/**
	 * Constructor with image type, image width, image height, and identifier reference.
	 * @param imageType image type.
	 * @param imageWidth image width.
	 * @param imageHeight image height.
	 * @param idRef identifier reference.
	 */
	public ConvVAEImpl(ImageType imageType, int imageWidth, int imageHeight, Id idRef) {
		super(toNeuronChannel(imageType), toActivationRef(imageType), idRef);
		
		this.imageType = imageType;
		this.imageWidth = imageWidth;
		this.imageHeight = imageHeight;
	}

	
	/**
	 * Constructor with image type, image width, and image height.
	 * @param imageType image type.
	 * @param imageWidth image width.
	 * @param imageHeight image height.
	 */
	public ConvVAEImpl(ImageType imageType, int imageWidth, int imageHeight) {
		this(imageType, imageWidth, imageHeight, null);
	}

	
	/**
	 * Initialize with Z dimension as well as hidden neurons.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode) {
		int xDim = imageWidth * imageHeight;
		return super.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode);
	}
	
	
	/**
	 * Initialize with Z dimension as well as hidden neurons.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode) {
		int xDim = imageWidth * imageHeight;
		return super.initialize(xDim, zDim, nHiddenNeuronEncode);
	}

	
	@Override
	public synchronized NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		for (Record record : sample) {
			try {
				if ((record.undefinedInput == null) || !(record.undefinedInput instanceof BufferedImage)) continue;
				
				BufferedImage image = (BufferedImage)record.undefinedInput;
				NeuronValue[] input = extractFromImage(image);
				if (input != null) record.input = input;
			}
			catch (Throwable e) {
				Util.trace(e);
			}
		}
		
		return super.learn(sample);
	}


	/**
	 * Generate values (X values).
	 * @return generated values.
	 * @throws RemoteException if any error raises.
	 */
	public BufferedImage generateImage() {
		try {
			NeuronValue[] dataX = generate();
			return convertToImage(dataX);
			
		} catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;
	}
	
	
	/**
	 * Extracting image into neuron value array.
	 * @param image specific image.
	 * @return neuron value array.
	 */
	protected NeuronValue[] extractFromImage(BufferedImage image) {
		if (image == null || imageWidth <= 0 || imageHeight <= 0) return null;
		
		if (SOURCE_RESIZING && image.getWidth() != imageWidth && image.getHeight() != imageHeight) {
			image = resizeImage(image, imageWidth, imageHeight);
			if (image == null) return null;
		}
		
		if (image.getType() != DEFAULT_SOURCE_IMAGE_TYPE) {
			image = convertToDefaultImage(image);
			if (image == null) return null;
		}
		
		if (image.getWidth() <= 0 && image.getHeight() <= 0) return null;

		NeuronValue[] values = new NeuronValue[imageWidth*imageHeight];

		double factor = IS_NORM ? 255 : 1;
		int minWidth = Math.min(imageWidth, image.getWidth());
		int minHeight = Math.min(imageHeight, image.getHeight());
		for (int y = 0; y < minHeight; y++) {
			for (int x = 0; x < minWidth; x++) {
				int p = image.getRGB(x, y);
				  
                int a = (p >> 24) & 0xff;
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;
                
                //Gray value
                int gray = (r + g + b) / 3;
  
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
  
                values[y*imageWidth + x] = value;
			}
			
		}
		
		//Fill zero.
		for (int y = minHeight; y < imageHeight; y++) {
			for (int x = minWidth; x < imageWidth; x++) {
	            NeuronValue value = null;
	            switch (imageType) {
	            case GRAY:
	            	value = new NeuronValue1(0).zero();
	            	break;
	            case RGB:
	            	value = new NeuronValueV(0, 0, 0).zero();
	            	break;
	            case ARGB:
	            	value = new NeuronValue1(0).zero();
	            	break;
	            default:
	            	value = new NeuronValue1(0).zero();
	            	break;
	            }
	            
                values[y*imageWidth + x] = value;
			}
		}
		
		return values;
	}
	
	
	/**
	 * Reading image into neuron value array.
	 * @param image specific image.
	 * @return neuron value array.
	 */
	protected NeuronValue[] read(Path imagePath) {
		try {
			InputStream is = Files.newInputStream(imagePath);
			BufferedImage image = ImageIO.read(is);
			NeuronValue[] values = extractFromImage(image);
			is.close();
			
			return values;
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;
	}
	
	
	/**
	 * Converting neuron values to image.
	 * @param values neuron values.
	 * @return converted image.
	 */
	protected BufferedImage convertToImage(NeuronValue[] values) {
		if (values == null || values.length == 0 || imageWidth <= 0 || imageHeight <= 0) return null;
		
		BufferedImage image = new BufferedImage(imageWidth, imageHeight, DEFAULT_SOURCE_IMAGE_TYPE);
		
		double factor = IS_NORM ? 255 : 1;
		for (int y = 0; y < imageHeight; y++) {
			for (int x = 0; x < imageWidth; x++) {
				int a = DEFAULT_ALPHA, r = 0, g = 0, b = 0, gray = 0;
				
				int index = y*imageWidth + x;
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
	 * Writing neuron value array to image.
	 * @param values neuron values.
	 * @param imagePath image path.
	 * @return true if writing is successful.
	 */
	protected boolean write(NeuronValue[] values, Path imagePath) {
		try {
			BufferedImage image = convertToImage(values);
			if (image == null) return false;
			
			OutputStream os = Files.newOutputStream(imagePath, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
			ImageIO.write(image, "png", os);
			os.close();
			
			return true;
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return false;
	}
	
	
	/**
	 * Resize image.
	 * @param image specific image.
	 * @return resized image.
	 */
	private static BufferedImage resizeImage(BufferedImage image, int newWidth, int newHeight) {
		if (image == null || newWidth <= 0 || newHeight <= 0)
			return null;
		else if (image.getWidth() != newWidth || image.getHeight() != newHeight) {
			Image resizedImage = image.getScaledInstance(newWidth, newHeight, Image.SCALE_DEFAULT);
			if (resizedImage == null) return null;
			
			image = convertToDefaultImage(resizedImage);
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
	 * Converting image to buffered image. The code is available at https://stackoverflow.com/questions/13605248/java-converting-image-to-bufferedimage 
	 * @param image specific image.
	 * @return buffered image.
	 */
	private static BufferedImage convertToDefaultImage(Image image) {
		if (image == null) return null;
		if (image instanceof BufferedImage) {
			BufferedImage bufferedImage = (BufferedImage)image;
			if (bufferedImage.getType() == DEFAULT_SOURCE_IMAGE_TYPE) return bufferedImage;
		}

		BufferedImage bufferedImage = new BufferedImage(image.getWidth(null), image.getHeight(null), DEFAULT_SOURCE_IMAGE_TYPE);
	    Graphics2D g = bufferedImage.createGraphics();
	    g.drawImage(image, 0, 0, null);
	    g.dispose();

	    return bufferedImage;
	}
	
	
	/**
	 * Converting image type to neuron channel. 
	 * @param imageType image type.
	 * @return neuron channel
	 */
	private static int toNeuronChannel(ImageType imageType) {
		int channel = 1;
        switch (imageType) {
        case GRAY:
        	channel = 1;
        	break;
        case RGB:
        	channel = 3;
        	break;
        case ARGB:
        	channel = 4;
        	break;
        default:
        	channel = 1;
        	break;
        }
        
        return channel;
	}
	
	
	/**
	 * Converting image type to neuron channel. 
	 * @param imageType image type.
	 * @return neuron channel
	 */
	private static Function toActivationRef(ImageType imageType) {
		Function f = null;
        switch (imageType) {
        case GRAY:
        	f = IS_NORM ? new LogisticFunction1(0.0, 1.0) : new LogisticFunction1(0.0, 255.0);
        	break;
        case RGB:
        	f = IS_NORM ? new LogisticFunctionV(3, 0.0, 1.0) : new LogisticFunctionV(3, 0.0, 255.0);
        	break;
        case ARGB:
        	f = IS_NORM ? new LogisticFunctionV(4, 0.0, 1.0) : new LogisticFunctionV(4, 0.0, 255.0);
        	break;
        default:
        	f = IS_NORM ? new LogisticFunction1(0.0, 1.0) : new LogisticFunction1(0.0, 255.0);
        	break;
        }
        
        return f;
	}


	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		try (ConvVAEImpl convVAE = new ConvVAEImpl(ImageType.RGB, 40, 40)) {
			convVAE.initialize(10, new int[] {30, 20});
		
			List<Record> sample = Util.newList(0);
			
			Record record = new Record();
			record.undefinedInput = convVAE.load(Paths.get("working/sample1.png"));
			sample.add(record);
			sample.add(record);
			sample.add(record);
			
			record = new Record();
			record.undefinedInput = convVAE.load(Paths.get("working/sample2.png"));
			sample.add(record);

			record = new Record();
			record.undefinedInput = convVAE.load(Paths.get("working/sample3.png"));
			sample.add(record);

			convVAE.learn(sample);
			
			//System.out.println(convVAE.toString());

			NeuronValue[] values = convVAE.generate();
			//NeuronValue[] values = convVAE.extractFromImage((BufferedImage)(record.undefinedInput));
			convVAE.write(values, Paths.get("working/gen.png"));
			
		}
		catch (Exception e) {
			Util.trace(e);
		}
	}


	/**
	 * Reading image into neuron value array.
	 * @param image specific image.
	 * @return neuron value array.
	 */
	private BufferedImage load(Path imagePath) {
		try {
			InputStream is = Files.newInputStream(imagePath);
			BufferedImage image = ImageIO.read(is);
			is.close();
			
			return image;
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;
	}


}
