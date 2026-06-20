/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.awt.image.RescaleOp;
import java.io.Serializable;
import java.util.Random;

import net.ea.ann.core.Util;

/**
 * This class implements augmentation operators.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Augmentor implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Augmentation operator.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	protected static enum Op {
		
		/**
		 * Flip.
		 */
		flip,
		
		/**
		 * Rotation.
		 */
		rotate,
		
		/**
		 * Resized cropping.
		 */
		crop,
		
		/**
		 * Color jittering.
		 */
		jitter,
		
		/**
		 * Gray-scaling.
		 */
		grayscale,
		
		/**
		 * Blur.
		 */
		blur,
		
		/**
		 * Sharpening.
		 */
		sharpen,
		
		/**
		 * Solarization.
		 */
		solarize,
		
		/**
		 * Erasing.
		 */
		erase,
		
	}
	
	
	/**
	 * Default list of augmentation operators.
	 */
	protected static Op[] operators = {
		Op.flip,
//		Op.rotate,
		Op.crop,
		Op.jitter,
		Op.grayscale,
		Op.blur,
		Op.sharpen,
		Op.solarize,
//		Op.erase
	};
	
	
	/**
	 * Internal raster.
	 */
	protected Raster raster = null;
	
	
	/**
	 * Constructor with raster.
	 * @param raster specified raster.
	 */
	public Augmentor(Raster raster) {
		this.raster = raster;
	}

	
	/**
	 * Creating random augmented raster.
	 * @return random augmented raster.
	 */
	public Raster augmentRandom() {
		if (!(this.raster instanceof Raster2D)) throw new IllegalArgumentException();
		Raster2D raster2D = (Raster2D)this.raster;
		Image image = raster2D.getImage();
		if (!(image instanceof ImageWrapper)) throw new IllegalArgumentException();
		
		//Cloning image.
		ImageWrapper imageWrapper = null;
		try {
			imageWrapper = (ImageWrapper) ((ImageWrapper)image).clone();
		} catch (Throwable e) {Util.trace(e);}
		
		//Creating random augmented image.
		BufferedImage augmented = augmentRandom(imageWrapper.getImage());
		Raster2DImpl augmentedRaster = new Raster2DImpl(new ImageWrapper(augmented));
		augmentedRaster.setProperty(this.raster.getProperty());
		return augmentedRaster;
	}
	
	
	/**
	 * Taking random augmentation operator.
	 * @param src source image.
	 * @return augmented version.
	 */
	private static BufferedImage augmentRandom(BufferedImage src) {
		int op = new Random().nextInt(operators.length);
		return augmentRandom(src, operators[op]);
	}
	
	
	/**
	 * Taking augmentation operator.
	 * @param src source image.
	 * @param op augmentation operator.
	 * @return augmented version.
	 */
	private static BufferedImage augmentRandom(BufferedImage src, Op op) {
		BufferedImage augmented = null;
		switch (op) {
		case flip:
			augmented = horizontalFlip(src);
			break;
		case rotate:
			augmented = rotateRandom(src);
			break;
		case crop:
			augmented = resizedCropRandom(src);
			break;
		case jitter:
			augmented = colorJitterRandom(src);
			break;
		case grayscale:
			augmented = grayScale(src);
			break;
		case blur:
			augmented = blur(src);
			break;
		case sharpen:
			augmented = sharpenRandom(src);
			break;
		case solarize:
			augmented = solarize(src);
			break;
		case erase:
			augmented = eraseRandom(src);
			break;
		default:
			augmented = src;
			break;
		}
		return augmented;
	}
	
	
	/**
	 * Horizontal flip
	 * @param src source image.
	 * @return horizontal flip image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage horizontalFlip(BufferedImage src) {
		int w = src.getWidth();
		int h = src.getHeight();
		BufferedImage dst = new BufferedImage(w, h, src.getType());
		Graphics2D g = dst.createGraphics();
		// Draw image mirrored: (x, y, w, h) -> (x+w, y, -w, h)
		g.drawImage(src, w, 0, -w, h, null);
		g.dispose();
		return dst;
	}


	/**
	 * Rotation.
	 * @param src source image.
	 * @param angle angle in degree.
	 * @return rotated image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage rotate(BufferedImage src, double angle) {
		double radians = Math.toRadians(angle);
		
		int w = src.getWidth();
		int h = src.getHeight();
		
		BufferedImage dst = new BufferedImage(w, h, src.getType());
		Graphics2D g = dst.createGraphics();
		
		//Set rotation around the center
		g.rotate(radians, w / 2.0, h / 2.0);
		g.drawImage(src, 0, 0, null);
		g.dispose();
		
		return dst;
	}

	
	/**
	 * Rotation.
	 * @param src source image.
	 * @param maxDegrees maximum angle in degree.
	 * @return rotated image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage rotateRandom(BufferedImage src, double maxDegrees) {
		double angle = (new Random().nextDouble() * 2 * maxDegrees) - maxDegrees;
		return rotate(src, angle);
	}
    
    
	/**
	 * Rotation.
	 * @param src source image.
	 * @return rotated image.
	 */
	private static BufferedImage rotateRandom(BufferedImage src) {return rotateRandom(src, 15.0);}
    
    
	/**
	 * Resized crop.
	 * @param src source image.
	 * @param targetW target width.
	 * @param targetH target height.
	 * @param scale minimum scale.
	 * @param x X coordinate.
	 * @param y Y coordinate
	 * @return resized cropped image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage resizedCrop(BufferedImage src, int targetW, int targetH, double scale, int x, int y) {
		int srcW = src.getWidth();
		int srcH = src.getHeight();
		int cropW = (int) (srcW * Math.sqrt(scale));
		int cropH = (int) (srcH * Math.sqrt(scale));
		
		BufferedImage cropped = src.getSubimage(x, y, cropW, cropH);
		
		//Resize back to target dimensions
		BufferedImage resized = new BufferedImage(targetW, targetH, src.getType());
		Graphics2D g = resized.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g.drawImage(cropped, 0, 0, targetW, targetH, null);
		g.dispose();
		
		return resized;
	}

	
	/**
	 * Resized crop.
	 * @param src source image.
	 * @param targetW target width.
	 * @param targetH target height.
	 * @param minScale minimum scale.
	 * @return resized cropped image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage resizedCropRandom(BufferedImage src, int targetW, int targetH, double minScale) {
		Random rand = new Random();
		
		int srcW = src.getWidth();
		int srcH = src.getHeight();
		
		double scale = minScale + (rand.nextDouble() * (1.0 - minScale));
		int cropW = (int) (srcW * Math.sqrt(scale));
		int cropH = (int) (srcH * Math.sqrt(scale));
		
		int x = rand.nextInt(Math.max(1, srcW - cropW));
		int y = rand.nextInt(Math.max(1, srcH - cropH));
		
		return resizedCrop(src, targetW, targetH, scale, x, y);
	}


	/**
	 * Resized crop with minimum scale 10%.
	 * @param src source image.
	 * @return resized cropped image.
	 */
	private static BufferedImage resizedCropRandom(BufferedImage src) {return resizedCropRandom(src, src.getWidth(), src.getHeight(), 0.1);}
    
    
	/**
	 * Color jitter (brightness and contrast).
	 * @param src source image.
	 * @param brightnessFactor brightness factor.
	 * @param contrastFactor contrast factor.
	 * @return image after color jitter operator.
	 * @author Gemini 2026.
	 */
	private static BufferedImage colorJitter(BufferedImage src, float brightnessFactor, float contrastFactor) {
		//RescaleOp applies: newPixel = (oldPixel * contrast) + (brightness_offset)
		//We use contrast factor as the scale and brightness factor (normalized) as the offset.
		//Factor = 1: neutral.
		RescaleOp op = new RescaleOp(contrastFactor, (brightnessFactor - 1.0f) * 128f, null);
		return op.filter(src, null);
	}
	
	
	/**
	 * Color jitter (brightness and contrast).
	 * @param src source image.
	 * @param brightness brightness.
	 * For instance, if it is 0.4 (40%), the brightness ranges in 0.6 (decreased 40%) and 1.4 (increased 40%).
	 * @param contrast contrast.
	 * For instance, if it is 0.2 (40%), the contrastion ranges in 0.8 (decreased 20%) and 1.2 (increased 20%).
	 * @return image after color jitter operator.
	 * @author Gemini 2026.
	 */
	private static BufferedImage colorJitterRandom(BufferedImage src, float brightness, float contrast) {
		Random rand = new Random();
		//Random factors: factor = 1.0 is neutral
		float bFactor = (1.0f - brightness) + (rand.nextFloat() * 2 * brightness);
		float cFactor = (1.0f - contrast) + (rand.nextFloat() * 2 * contrast);
		return colorJitter(src, bFactor, cFactor);
	}

    
	/**
	 * Color jitter.
	 * @param src source image.
	 * @return image after color jitter operator.
	 */
	private static BufferedImage colorJitterRandom(BufferedImage src) {return colorJitterRandom(src, 0.5f, 0.5f);}
	
	
	/**
	 * Random gray-scale.
	 * @param src source image.
	 * @return gray-scaled image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage grayScale(BufferedImage src) {
		ColorConvertOp op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null);
		BufferedImage gray = op.filter(src, null);
		
		//To keep 3 channels (RGB) for the CNN, draw gray image back onto an RGB buffer
		BufferedImage dst = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics2D g = dst.createGraphics();
		g.drawImage(gray, 0, 0, null);
		g.dispose();
		return dst;
	}

    
	/**
	 * Gaussian blur (simplified via ConvolveOp).
	 * @param src source image.
	 * @param radius radius of kernel size, kernel size = 2*radius + 1, for instance, radius=1 producing 3x3 kernel filter.
	 * @return blurred image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage blur(BufferedImage src, int radius) {
		int size = radius * 2 + 1;
		float weight = 1.0f / (size * size);
		float[] data = new float[size * size];
		for (int i = 0; i < data.length; i++) data[i] = weight;
		
		Kernel kernel = new Kernel(size, size, data);
		ConvolveOp op = new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null);
		return op.filter(src, null);
	}

    
	/**
	 * Gaussian blur.
	 * @param src source image.
	 * @return blurred image.
	 */
	private static BufferedImage blur(BufferedImage src) {return blur(src, 1);}
	
	
	/**
	 * Sharpness.
	 * @param src source image.
	 * @param sharpnessFactor
	 * @return sharpened image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage sharpen(BufferedImage src, float sharpnessFactor) {
		//sharpnessFactor: 1.0 = not changed, 1.0 = sharpen, 2.0 = crisp
		
		//Calculate the center and edge weights dynamically using the factor
		float edge = -sharpnessFactor;
		float center = 1.0f + (4.0f * sharpnessFactor);
		
		//A standard 3x3 sharpening matrix that responds to the factor
		float[] matrix = {
			0.0f,  edge,  0.0f,
			edge, center,  edge,
			0.0f,  edge,  0.0f
		};
		
		Kernel kernel = new Kernel(3, 3, matrix);
		
		//ConvolveOp applies the kernel to the image
	    ConvolveOp op = new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null);
	    return op.filter(src, null);
	}
    

	/**
	 * Sharpness.
	 * @param src
	 * @return sharpened image.
	 */
	private static BufferedImage sharpenRandom(BufferedImage src) {
		float minSharpness = 1.0f;
		float maxSharpness = 2.0f;
		float randomFactor = minSharpness + new Random().nextFloat() * (maxSharpness - minSharpness);
		return sharpen(src, randomFactor);
	}
	
	
	/**
	 * Solarization.
	 * @param src source image.
	 * @param threshold threshold from 0 to 255. If pixel value is larger than or equal to this threshold, it is converted.
	 * @return solarized image.
	 * @author Gemini 2026.
	 */
	private static BufferedImage solarize(BufferedImage src, int threshold) {
		int w = src.getWidth();
		int h = src.getHeight();
		BufferedImage dst = new BufferedImage(w, h, src.getType());
		
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int rgb = src.getRGB(x, y);
				int r = (rgb >> 16) & 0xFF;
				int g = (rgb >> 8) & 0xFF;
				int b = rgb & 0xFF;
				
				if (r >= threshold) r = 255 - r;
				if (g >= threshold) g = 255 - g;
				if (b >= threshold) b = 255 - b;
				
				dst.setRGB(x, y, (r << 16) | (g << 8) | b);
			}
		}
		return dst;
	}

	
	/**
	 * Solarization.
	 * @param srcsource image.
	 * @return solarized image.
	 */
	private static BufferedImage solarize(BufferedImage src) {
		return solarize(src, 128);
	}
	
	
	/**
	 * Erasing.
	 * @param src source image.
	 * @param minArea minimum percentage area is erased, for instance, minArea=0.02 means that at least 2% of the image area is erased.
	 * @param maxArea maximum percentage area is erased, for instance, minArea=0.08 means that at most 8% of the image area is erased.
	 * @return image whose some area is erased.
	 * @author Gemini 2026.
	 */
	private static BufferedImage eraseRandom(BufferedImage src, double minArea, double maxArea) {
		Random rand = new Random();
		
		int w = src.getWidth();
		int h = src.getHeight();
		int area = w * h;
		
		//Determine size of the "erase" rectangle.
		double targetArea = (minArea + rand.nextDouble() * (maxArea - minArea)) * area;
		double aspect = 0.3 + rand.nextDouble() * 3.0; //random aspect ratio.
		
		int eraseH = (int) Math.sqrt(targetArea * aspect);
		int eraseW = (int) Math.sqrt(targetArea / aspect);
		
		if (eraseW < w && eraseH < h) {
			int x = rand.nextInt(w - eraseW);
			int y = rand.nextInt(h - eraseH);
			
			Graphics2D g = src.createGraphics();
			g.setColor(new Color(128, 128, 128)); //neutral gray.
			g.fillRect(x, y, eraseW, eraseH);
			g.dispose();
		}
		return src;
	}
    

	/**
	 * Erasing.
	 * @param src source image.
	 * @return image whose some area is erased.
	 */
	private static BufferedImage eraseRandom(BufferedImage src) {return eraseRandom(src, 0.01, 0.08);}
	
	
}
