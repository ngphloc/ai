/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.raster.MedMNISTNpzLoader.Dataset;

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
	 * UCI optical digits name.
	 */
	public final static String UCI_OPT_DIGITS = "uciod";

	
	/**
	 * Medical MNIST in form of NPZ file.
	 */
	public final static String MNIST_MED_NPZ = "mnistmed";
	
	
	/**
	 * Number of CIFAR10 images.
	 */
	public final static int CIFAR10_NUMBER_IMAGES = 10000;
	
	
	/**
	 * Loading buffered image flag.
	 */
	private final static boolean CIFAR10_LOAD_BUFFERED_IMAGE = true;
	
	
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
					
					if (CIFAR10_LOAD_BUFFERED_IMAGE) {
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
					}
					else {
						ImageMatrix image = new ImageMatrix(new Size(width, height, 3), 1);
						Matrix[] matrices = MatrixUtil.split(image.data);
						int wh = width*height;
						for (int y = 0; y < height; y++) {
							int yw = y*width;
							for (int x = 0; x < width; x++) {
								double r = (double)imageData[yw + x] / 255.0;
								double g = (double)imageData[wh + yw + x] / 255.0;
								double b = (double)imageData[2*wh + yw + x] / 255.0;
								matrices[0].setv(y, x, r);
								matrices[1].setv(y, x, g);
								matrices[2].setv(y, x, b);
							}
						}
						labeledImages.add(new LabeledImage(image, label));
					}
					
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
	
	
	/**
	 * Loading UCI optical digits dataset.
	 * @param path path of UCI optical digits dataset.
	 * @return list of loaded images.
	 */
	public static List<LabeledImage> loadUCIOptDigits(Path path) {
		List<LabeledImage> labeledImages = Util.newList(0);
		if (path == null) return labeledImages;

		try (BufferedReader reader = Files.newBufferedReader(path)) {
		    String line = null;
		    while ((line = reader.readLine()) != null) {
				line = line.trim();
				if (line.isEmpty()) continue;
				String[] tokens = line.split(",");
				if (tokens.length < 65) continue; //Each row is image vector of 64 = 8x8 pixels plus 1 label.
				
				ImageMatrix image = new ImageMatrix(new Size(8, 8), 1);
				Matrix matrix = image.get();
				
				//Parse the first 64 values into an 8x8 spatial grid. Max value in UCI is 16, divide by 16.0 to normalize [0, 1].
				for (int i = 0; i < 64; i++) {
					int row = i / 8;
					int col = i % 8;
					try {
						double value = Double.parseDouble(tokens[i]) / 16.0;
						matrix.setv(row, col, value);
					} catch (Throwable e) {Util.trace(e);}
				}
				    
				int label = Integer.parseInt(tokens[64]); //The last token is the label (0-9).
				labeledImages.add(new LabeledImage(image, label));
		    }
		} catch (Throwable e) {Util.trace(e);}
		
		return labeledImages;
	}
	
	
	/**
	 * Loading medical MNIST dataset in form of NPZ file.
	 * @param path Path to the .npz file (e.g., "pneumoniamnist.npz").
	 * @param split split.
	 * @return list of loaded images.
	 */
	public static List<LabeledImage> loadNpzMedMNIST(Path path, String split) {
		List<LabeledImage> labeledImages = Util.newList(0);
		try {
			Dataset dataset = MedMNISTNpzLoader.loadSplit(path, split);
			for (int i = 0; i < dataset.images.length; i++) {
				double[][][] shape = dataset.images[i];
				int channels = shape.length;
				int height = shape[0].length;
				int width = shape[0][0].length;
				
				ImageMatrix image = new ImageMatrix(new Size(width, height, channels), 1);
				Matrix[] matrices = MatrixUtil.split(image.get());
				for (int c = 0; c < channels; c++) {
					for (int h = 0; h < height; h++) {
						for (int w = 0; w < width; w++) {
							matrices[c].setv(h, w, shape[c][h][w]);
						}
					}
				}

				labeledImages.add(new LabeledImage(image, dataset.labels[i]));
			}
		} catch (Throwable e) {Util.trace(e);}
		
		return labeledImages;
	}
	
	
}



/**
 * This class loads medical MNIST dataset in form of NPZ file. 
 * @author Gemini
 * @version 1.0
 *
 */
class MedMNISTNpzLoader {

	
	/**
	 * This class represents medical MNIST dataset
	 * @author Gemini
	 * @version 1.0
	 *
	 */
    public static class Dataset implements Cloneable, Serializable {

    	/**
    	 * Serial version UID for serializable class. 
    	 */
    	private static final long serialVersionUID = 1L;
    	
    	/**
    	 * Internal images.
    	 */
        public final double[][][][] images; // [Batch][Channels][Height][Width] normalized to [0.0, 1.0]
        
        /**
         * Internal label.
         */
        public final int[] labels;          // [Batch] target class indices

        
        /**
         * Constructor with images and labels.
         * @param images images.
         * @param labels labels.
         */
        public Dataset(double[][][][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
        
    }
    

    /**
     * Loads images and labels for a specific split ('train', 'val', or 'test') from a MedMNIST .npz file.
     * @param npzFilePath Path to the .npz file (e.g., "pneumoniamnist.npz").
     * @param split Split prefix: "train", "val", or "test".
     * @return Dataset containing normalized 4D image array and label vector.
     * @throws IO exception if IO error raises.
     */
    public static Dataset loadSplit(Path npzFilePath, String split) throws Exception {
		String imageEntryName = split + "_images.npy";
		String labelEntryName = split + "_labels.npy";
		
		byte[] imageBytes = extractEntry(npzFilePath, imageEntryName);
		byte[] labelBytes = extractEntry(npzFilePath, labelEntryName);
		
		if (imageBytes == null || labelBytes == null) {
			throw new IllegalArgumentException("Could not find entries for split '" + split + "' in " + npzFilePath);
		}
		
		double[][][][] images = parseNpyImages(imageBytes);
		int[] labels = parseNpyLabels(labelBytes);
		
		return new Dataset(images, labels);
    }

    
    /**
     * Extracting entry in NPZ file.
     * @param npzFilePath NPZ file path.
     * @param targetEntry entry.
     * @return data in array of 
     * @throws Exception if error raises.
     */
    private static byte[] extractEntry(Path npzFilePath, String targetEntry) throws Exception {
		// Uses Files.newInputStream(npzFilePath) instead of FileInputStream
		try (InputStream fis = Files.newInputStream(npzFilePath);
			ZipInputStream zis = new ZipInputStream(new BufferedInputStream(fis))) {
			
			ZipEntry entry;
			while ((entry = zis.getNextEntry()) != null) {
				if (entry.getName().equals(targetEntry)) {
					return zis.readAllBytes();
				}
			}
		}
		return null;
    }

    
    /**
     * Parses a NumPy .npy array of image bytes (uint8) into a normalized 4D Java double array. This is to read a shape.
     * @param npyBytes NumPy .npy array of image bytes (uint8).
     * @return normalized 4D Java double array.
     */
    private static double[][][][] parseNpyImages(byte[] npyBytes) {
		ByteBuffer buffer = ByteBuffer.wrap(npyBytes).order(ByteOrder.LITTLE_ENDIAN);
		
		// 1. Verify Magic Header: \x93NUMPY
		byte magic0 = buffer.get();
		byte[] magicBytes = new byte[5];
		buffer.get(magicBytes);
		String magicStr = new String(magicBytes, StandardCharsets.US_ASCII);
		
		if ((magic0 & 0xFF) != 0x93 || !"NUMPY".equals(magicStr)) {
			throw new IllegalArgumentException("Invalid .npy file header format.");
		}
		
		// Skip Major & Minor version bytes
		buffer.get(); 
		buffer.get(); 
		
		// 2. Scan for the ending newline byte '\n' (0x0A)
		int headerStartPos = buffer.position();
		int newlinePos = -1;
		for (int i = headerStartPos; i < npyBytes.length; i++) {
			if (npyBytes[i] == 0x0A) { // '\n'
				newlinePos = i;
				break;
			}
		}
		
		if (newlinePos == -1) {
			throw new IllegalArgumentException("Failed to locate end of NPY header.");
		}
		
		String rawHeaderStr = new String(npyBytes, headerStartPos, newlinePos - headerStartPos, StandardCharsets.US_ASCII);
		buffer.position(newlinePos + 1);
		
		// 3. Parse shape
		int[] shape = parseHeaderShape(rawHeaderStr);
		
		int N = shape[0];
		int H = shape[1];
		int W = shape[2];
		int C = (shape.length == 4) ? shape[3] : 1;
		
		double[][][][] images = new double[N][C][H][W];
		
		// 4. Read raw uint8 pixel bytes directly into 4D double array
		for (int i = 0; i < N; i++) {
			if (shape.length == 4) { // RGB / Multi-channel: [N, H, W, C]
				for (int h = 0; h < H; h++) {
					for (int w = 0; w < W; w++) {
						for (int c = 0; c < C; c++) {
							int pixel = buffer.get() & 0xFF;
							images[i][c][h][w] = pixel / 255.0;
						}
					}
				}
			} else { // Grayscale: [N, H, W]
				for (int h = 0; h < H; h++) {
					for (int w = 0; w < W; w++) {
						int pixel = buffer.get() & 0xFF;
						images[i][0][h][w] = pixel / 255.0;
					}
				}
			}
		}
		
		return images;
    }

    
    /**
     * Parses a NumPy .npy array of target label bytes (uint8 / int64) into a 1D Java int array.
     * @param npyBytes NumPy .npy array of target label bytes (uint8 / int64).
     * @return 1D Java int array.
     */
    private static int[] parseNpyLabels(byte[] npyBytes) {
		ByteBuffer buffer = ByteBuffer.wrap(npyBytes).order(ByteOrder.LITTLE_ENDIAN);
		buffer.position(8);
		
		int newlinePos = -1;
		for (int i = 8; i < npyBytes.length; i++) {
			if (npyBytes[i] == 0x0A) {
				newlinePos = i;
				break;
			}
		}
		
		String rawHeaderStr = new String(npyBytes, 8, newlinePos - 8, StandardCharsets.US_ASCII);
		buffer.position(newlinePos + 1);
		
		int[] shape = parseHeaderShape(rawHeaderStr);
		int totalLabels = shape[0];
		int[] labels = new int[totalLabels];
		
		for (int i = 0; i < totalLabels; i++) {
			if (rawHeaderStr.contains("u1") || rawHeaderStr.contains("i1") || rawHeaderStr.contains("|u1")) {
				labels[i] = buffer.get() & 0xFF;
			} else if (rawHeaderStr.contains("i8") || rawHeaderStr.contains("u8") || rawHeaderStr.contains("<i8")) {
				labels[i] = (int) buffer.getLong();
			} else {
				labels[i] = buffer.getInt();
			}
		}
		
		return labels;
    }

    
    /**
     * Parsing header of a shape.
     * @param header header.
     * @return parsed array.
     */
    private static int[] parseHeaderShape(String header) {
		int start = header.indexOf('(') + 1;
		int end = header.indexOf(')');
		String shapeStr = header.substring(start, end).replace(" ", "");
		String[] parts = shapeStr.split(",");
		
		int count = 0;
		for (String p : parts) {
			if (!p.isEmpty()) count++;
		}
		
		int[] shape = new int[count];
		int idx = 0;
		for (String p : parts) {
			if (!p.isEmpty()) {
				shape[idx++] = Integer.parseInt(p);
			}
		}
		return shape;
    }
    
    
    /**
     * Reading ASCII string.
     * @param buffer buffer.
     * @param length lengh to read.
     * @return ASCII string.
     */
    @SuppressWarnings("unused")
    @Deprecated
	private static String readAsciiString(ByteBuffer buffer, int length) {
		byte[] bytes = new byte[length];
		buffer.get(bytes);
		return new String(bytes, StandardCharsets.US_ASCII);
    }
    
    
}


