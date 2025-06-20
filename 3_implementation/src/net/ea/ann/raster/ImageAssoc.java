/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import net.ea.ann.core.Util;

/**
 * This class is associator of image.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ImageAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * CIFAR name.
	 */
	public final static String CIFAR = "cifar";
	
	
	/**
	 * Number of CIFAR10 images.
	 */
	public final static int CIFAR10_NUMBER_IMAGES = 10000;
	
	
	/**
	 * Internal image.
	 */
	protected Image image = null;
	
	
	/**
	 * Constructor with image.
	 * @param image specified image.
	 */
	public ImageAssoc(Image image) {
		this.image = image;
	}

	
	/**
	 * This class represent labeled image.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class LabeledImage implements Cloneable, Serializable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Stored image.
		 */
		public Image image = null;
		
		/**
		 * Image label.
		 */
		public int label = 0;

		/**
		 * Constructor with image and label.
		 * @param image specified image.
		 * @param label specified label.
		 */
		public LabeledImage(Image image, int label) {
			this.image = image;
			this.label = label;
		}
		
		/**
		 * Converting this labeled image to raster.
		 * @return raster converted from this labeled image.
		 */
		public Raster2D toRaster() {
			if (image == null) return null;
			Raster2DImpl raster = new Raster2DImpl(image);
			raster.getProperty().setLabelId(label);
			return raster;
		}
		
	}
	
	
	/**
	 * Loading CIFAR-10 dataset.
	 * @param path path of CIFAR-10 dataset.
	 * @param nImages number of images to be loaded.
	 * @return list of loaded images.
	 */
	public static List<LabeledImage> loadCIFAR10(Path path, int nImages) {
		List<LabeledImage> labeledImages = Util.newList(0);
		if (path == null) return labeledImages;
		
		nImages = nImages <= 0 ? CIFAR10_NUMBER_IMAGES : nImages;
		nImages = Math.min(nImages, CIFAR10_NUMBER_IMAGES);
		try (InputStream is = Files.newInputStream(path)) {
			int width = 32, height = 32;
			int size = 32*32*3;
			for (int i = 0; i < nImages; i++) {
				try {
					int label = is.read();
					if (label < 0) break;
					
					byte[] imageData = new byte[size];
					if (is.read(imageData) < 0) break;
					BufferedImage image = new BufferedImage(width, height, Image.SOURCE_IMAGE_TYPE_DEFAULT);
					int wh = width*height;
					for (int y = 0; y < height; y++) {
						int yw = y*width;
						for (int x = 0; x < width; x++) {
							int a = Image.ALPHA_DEFAULT;
							int r = imageData[yw + x];
							int g = imageData[wh + yw + x];
							int b = imageData[2*wh + yw + x];
							int color = (a << 24) | (r << 16) | (g << 8) | b;
							image.setRGB(x, y, color);
						}
					}
					
					ImageWrapper imageWrapper = new ImageWrapper(image);
					labeledImages.add(new LabeledImage(imageWrapper, label));
				} catch (Throwable e) {Util.trace(e);}
			}
		}
		catch (Throwable e) {Util.trace(e);}

		return labeledImages;
	}
	
	
	/**
	 * Loading CIFAR-10 dataset.
	 * @param path path of CIFAR-10 dataset.
	 * @return list of loaded images.
	 */
	public static List<LabeledImage> loadCIFAR10(Path path) {
		return loadCIFAR10(path, -1);
	}
	
	
}
