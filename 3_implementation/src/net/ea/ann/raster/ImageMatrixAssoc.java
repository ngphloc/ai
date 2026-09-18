/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.io.Serializable;
import java.util.Random;

/**
 * This utility class provides utility methods to process matrix image.
 * @author USER
 *
 */
public class ImageMatrixAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal image.
	 */
	protected ImageMatrix image = null;
	
	
	/**
	 * Constructor with image.
	 * @param image image.
	 */
	public ImageMatrixAssoc(ImageMatrix image) {
		this.image = image;
	}

	
	/**
	 * Horizontal flip.
	 * @param src source image.
	 * @return horizontal flip image. 
	 */
	private static double[][][] horizontalFlip(double[][][] src) {
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		double[][][] dst = new double[channels][height][width];

		for (int c = 0; c < channels; c++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) dst[c][y][x] = src[c][y][width - 1 - x];
			}
		}
        return dst;
    }
	

	/**
	 * Horizontal flip.
	 * @return horizontal flip image. 
	 */
	public ImageMatrix horizontalFlip() {
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(horizontalFlip(src)) : null;
	}
	
	
	/**
	 * Rotation.
	 * @param src source image.
	 * @param angle angle in degree.
	 * @return rotated image.
	 * @author Gemini 2026
	 */
	private static double[][][] rotate(double[][][] src, double angle) {
		double rad = Math.toRadians(-angle); //Reverse angle for backward mapping
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		
		double cX = (width - 1) / 2.0;
		double cY = (height - 1) / 2.0;
		double cos = Math.cos(rad);
		double sin = Math.sin(rad);
		double[][][] dst = new double[channels][height][width];
		
		for (int dstY = 0; dstY < height; dstY++) {
			for (int dstX = 0; dstX < width; dstX++) {
				double xOff = dstX - cX;
				double yOff = dstY - cY;
				
				double srcX = cX + (xOff * cos - yOff * sin);
				double srcY = cY + (xOff * sin + yOff * cos);
				
				int x0 = (int) Math.floor(srcX);
				int y0 = (int) Math.floor(srcY);
				
				if (x0 >= 0 && x0 < width - 1 && y0 >= 0 && y0 < height - 1) {
					double xLerp = srcX - x0;
					double yLerp = srcY - y0;
					
					for (int c = 0; c < channels; c++) {
						double top = src[c][y0][x0] * (1 - xLerp) + src[c][y0][x0 + 1] * xLerp;
						double bottom = src[c][y0 + 1][x0] * (1 - xLerp) + src[c][y0 + 1][x0 + 1] * xLerp;
						dst[c][dstY][dstX] = top * (1 - yLerp) + bottom * yLerp;
					}
				}
			}
		}
        return dst;
    }
	

	/**
	 * Rotation.
	 * @param src source image.
	 * @param maxDegrees maximum angle in degree.
	 * @return rotated image.
	 * @author Chat-GPT 2026
	 */
	private static double[][][] rotateRandom(double[][][] src, double maxDegrees) {
		double angle = (new Random().nextDouble() * 2 * maxDegrees) - maxDegrees;
		return rotate(src, angle);
	}

	
	/**
	 * Rotation.
	 * @param src source image.
	 * @return rotated image.
	 */
	private static double[][][] rotateRandom(double[][][] src) {return rotateRandom(src, 20.0);}

	
	/**
	 * Random rotation.
	 * @return rotated image. 
	 */
	public ImageMatrix rotateRandom() {
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(rotateRandom(src)) : null;
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
    private static double[][][] resizedCropRandom(double[][][] src, int targetW, int targetH, double minScale) {
		int channels = src.length, srcH = src[0].length, srcW = src[0][0].length;
		
		Random rand = new Random();
		double scale = minScale + (rand.nextDouble() * (1.0 - minScale));
		int cropW = Math.max(1, (int) (srcW * Math.sqrt(scale)));
		int cropH = Math.max(1, (int) (srcH * Math.sqrt(scale)));
		
		int startX = rand.nextInt(Math.max(1, srcW - cropW));
		int startY = rand.nextInt(Math.max(1, srcH - cropH));
		
		double[][][] dst = new double[channels][targetH][targetW];
		
		//Bilinear Interpolation from Cropped Region to Target Dimensions
		for (int c = 0; c < channels; c++) {
			for (int dstY = 0; dstY < targetH; dstY++) {
				double srcY = startY + (dstY * (double) cropH / targetH);
				int y0 = (int) Math.floor(srcY);
				int y1 = Math.min(y0 + 1, srcH - 1);
				double yLerp = srcY - y0;
				
				for (int dstX = 0; dstX < targetW; dstX++) {
					double srcX = startX + (dstX * (double) cropW / targetW);
					int x0 = (int) Math.floor(srcX);
					int x1 = Math.min(x0 + 1, srcW - 1);
					double xLerp = srcX - x0;
					
					double top = src[c][y0][x0] * (1 - xLerp) + src[c][y0][x1] * xLerp;
					double bottom = src[c][y1][x0] * (1 - xLerp) + src[c][y1][x1] * xLerp;
					
					dst[c][dstY][dstX] = top * (1 - yLerp) + bottom * yLerp;
				}
			}
		}
		return dst;
    }

    
	/**
	 * Resized crop with minimum scale 10%.
	 * @param src source image.
	 * @return resized cropped image.
	 */
	private static double[][][] resizedCropRandom(double[][][] src) {
        int srcH = src[0].length, srcW = src[0][0].length;
		return resizedCropRandom(src, srcW, srcH, 0.2);
	}
	
	
	/**
	 * Resized crop.
	 * @return resized cropped image.
	 */
	public ImageMatrix resizedCropRandom() {
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(resizedCropRandom(src)) : null;
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
    private static double[][][] colorJitterRandom(double[][][] src, double brightness, double contrast) {
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		
		Random rand = new Random();
		double bFactor = (1.0 - brightness) + (rand.nextDouble() * 2 * brightness);
		double cFactor = (1.0 - contrast) + (rand.nextDouble() * 2 * contrast);
		
		double[][][] dst = new double[channels][height][width];
		
		//Rescaling math: newPixel = (oldPixel - 0.5) * cFactor + 0.5 * bFactor
		//Works seamlessly for pixel ranges normalized to [0, 1] or [0, 255]
		for (int c = 0; c < channels; c++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					double val = src[c][y][x];
					double rescaled = (val - 0.5) * cFactor + (0.5 * bFactor);
					dst[c][y][x] = Math.max(0.0, Math.min(1.0, rescaled));
				}
			}
		}
		return dst;
    }


	/**
	 * Color jitter.
	 * @param src source image.
	 * @return image after color jitter operator.
	 */
	private static double[][][] colorJitterRandom(double[][][] src) {
		return colorJitterRandom(src, 0.4, 0.4);
	}


	/**
	 * Resized crop.
	 * @return image after color jitter operator.
	 */
	public ImageMatrix colorJitterRandom() {
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(colorJitterRandom(src)) : null;
	}


	/**
	 * Gray-scale.
	 * @param src source image.
	 * @return gray-scaled image.
	 * @author Gemini 2026.
	 */
	private static double[][][] grayScale(double[][][] src) {
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		assert (channels == 3);
		
		double[][][] dst = new double[channels][height][width];
		
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				//Luminosity formula: 0.299*R + 0.587*G + 0.114*B
				double gray = (0.299 * src[0][y][x]) + (0.587 * src[1][y][x]) + (0.114 * src[2][y][x]);
				
				//Assign gray value across all channels to retain 3D shape
				for (int c = 0; c < channels; c++) dst[c][y][x] = gray;
			}
		}
		return dst;
	}


	/**
	 * Gray-scale.
	 * @return gray-scaled image.
	 */
	public ImageMatrix grayScale() {
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(grayScale(src)) : null;
	}


	/**
	 * Gaussian blur.
	 * @param src source image.
	 * @param radius radius of kernel size, kernel size = 2*radius + 1, for instance, radius=1 producing 3x3 kernel filter.
	 * @return blurred image.
	 * @author Gemini 2026.
	 */
	private static double[][][] blur(double[][][] src, int radius) {
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		
		int kSize = radius * 2 + 1;
		double sigma = radius / 3.0;
		double[][] kernel = generateGaussianKernel(kSize, sigma);
		
		double[][][] dst = new double[channels][height][width];
		int pad = kSize / 2;
		
		for (int c = 0; c < channels; c++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					double sum = 0.0;
					for (int ky = -pad; ky <= pad; ky++) {
						for (int kx = -pad; kx <= pad; kx++) {
							int py = Math.min(Math.max(y + ky, 0), height - 1);
							int px = Math.min(Math.max(x + kx, 0), width - 1);
							sum += src[c][py][px] * kernel[ky + pad][kx + pad];
						}
					}
					dst[c][y][x] = sum;
				}
			}
		}
		return dst;
	}

    
	/**
	 * Helper method to create a 2D Gaussian Kernel.
	 * @param size kernel size.
	 * @param sigma sigma.
	 * @return kernel.
	 * @author Gemini 2026.
	 */
	private static double[][] generateGaussianKernel(int size, double sigma) {
		double[][] kernel = new double[size][size];
		double sum = 0.0;
		int center = size / 2;
		
		for (int y = 0; y < size; y++) {
			for (int x = 0; x < size; x++) {
				int dx = x - center;
				int dy = y - center;
				double val = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma);
				kernel[y][x] = val;
				sum += val;
			}
		}
		//Normalize Kernel
		for (int y = 0; y < size; y++) {
			for (int x = 0; x < size; x++) kernel[y][x] /= sum;
		}
		return kernel;
	}


	/**
	 * Gaussian blur.
	 * @return blurred image.
	 */
	public ImageMatrix blur() {
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(blur(src, 3)) : null;
	}


	/**
	 * Sharpening image.
	 * @param src source image.
	 * @param sharpnessFactor
	 * @return sharpened image.
	 * @author Gemini 2026.
	 */
    private static double[][][] sharpen(double[][][] src, double sharpnessFactor) {
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		
		double edge = -sharpnessFactor;
		double center = 1.0 + (4.0 * sharpnessFactor);
		double[][] kernel = {
			{ 0.0,  edge,  0.0 },
			{ edge, center, edge },
			{ 0.0,  edge,  0.0 }
		};
		
		double[][][] dst = new double[channels][height][width];
		
		for (int c = 0; c < channels; c++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					double sum = 0.0;
					for (int ky = -1; ky <= 1; ky++) {
						for (int kx = -1; kx <= 1; kx++) {
							int py = Math.min(Math.max(y + ky, 0), height - 1);
							int px = Math.min(Math.max(x + kx, 0), width - 1);
							sum += src[c][py][px] * kernel[ky + 1][kx + 1];
						}
					}
					dst[c][y][x] = Math.max(0.0, Math.min(1.0, sum));
				}
			}
		}
		return dst;
    }


	/**
	 * Sharpening image.
	 * @param src
	 * @return sharpened image.
	 */
	public ImageMatrix sharpenRandom() {
		double minSharpness = 0.1;
		double maxSharpness = 2.0;
		double randomFactor = minSharpness + new Random().nextDouble() * (maxSharpness - minSharpness);
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(sharpen(src, randomFactor)) : null;
	}


	/**
	 * Solarization.
	 * @param src source image.
	 * @param threshold threshold from 0 to 1. If pixel value is larger than or equal to this threshold, it is converted.
	 * @return solarized image.
	 * @author Gemini 2026.
	 */
	private static double[][][] solarize(double[][][] src, double threshold) {
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		
		//Auto-detect max domain scale (e.g., 1.0 or 255.0)
		double maxVal = (threshold <= 1.0) ? 1.0 : 1.0; //255.0;
		double[][][] dst = new double[channels][height][width];
		
		for (int c = 0; c < channels; c++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					double val = src[c][y][x];
					if (val >= threshold)
						dst[c][y][x] = maxVal - val;
					else
						dst[c][y][x] = val;
				}
			}
		}
		return dst;
	}


	/**
	 * Solarization.
	 * @param srcsource image.
	 * @return solarized image.
	 */
	public ImageMatrix solarize() {
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(solarize(src, 0.5)) : null;
	}


	/**
	 * Erasing.
	 * @param src source image.
	 * @param minArea minimum percentage area is erased, for instance, minArea=0.02 means that at least 2% of the image area is erased.
	 * @param maxArea maximum percentage area is erased, for instance, minArea=0.08 means that at most 8% of the image area is erased.
	 * @param fillValue filling value.
	 * @return image whose some area is erased.
	 * @author Gemini 2026.
	 */
	private static double[][][] eraseRandom(double[][][] src, double minArea, double maxArea, double fillValue) {
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		int totalArea = height * width;
		
		Random rand = new Random();
		double targetArea = (minArea + rand.nextDouble() * (maxArea - minArea)) * totalArea;
		double aspect = 0.3 + rand.nextDouble() * 2.7;
		
		int eraseH = Math.max(1, (int) Math.sqrt(targetArea * aspect));
		int eraseW = Math.max(1, (int) Math.sqrt(targetArea / aspect));
		
		if (eraseW < width && eraseH < height) {
			int startX = rand.nextInt(width - eraseW);
			int startY = rand.nextInt(height - eraseH);
			
			double[][][] dst = cloneArray(src);
			
			for (int c = 0; c < channels; c++) {
				for (int y = startY; y < startY + eraseH; y++) {
					for (int x = startX; x < startX + eraseW; x++) {
						dst[c][y][x] = fillValue;
					}
				}
			}
			return dst;
		}
		return src;
	}


	/**
	 * Erasing.
	 * @param src source image.
	 * @return image whose some area is erased.
	 */
	public ImageMatrix eraseRandom() {
		double[][][] src = this.image.getReals();
		return src != null ? new ImageMatrix(eraseRandom(src, 0.02, 0.2, 0.5)) : null;
	}

	
	/**
	 * Helper method to deep clone 3D double arrays.
	 * @param src source arrays.
	 * @return cloned arrays.
	 */
	private static double[][][] cloneArray(double[][][] src) {
		int channels = src.length, height = src[0].length, width = src[0][0].length;
		
		double[][][] dst = new double[channels][height][width];
		for (int c = 0; c < channels; c++) {
			for (int y = 0; y < height; y++) {
				System.arraycopy(src[c][y], 0, dst[c][y], 0, width);
			}
		}
		return dst;
	}


}
