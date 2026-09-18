/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixReal;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.raster.Raster.RasterType;

/**
 * This class represents matrix image whose pixels range in interval [0, 1]. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ImageMatrix implements Image, Sound {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Image data.
	 */
	protected Matrix data =  null;
	
	
	/**
	 * Constructor with matrix.
	 * @param data image data.
	 */
	public ImageMatrix(Matrix data) {
		assert (data != null);
		this.data = data;
	}

	
	/**
	 * Constructor with matrix.
	 * @param data image data.
	 */
	public ImageMatrix(double[][] data) {
		assert (data != null);
		MatrixReal matrix = MatrixReal.Wrap(data);
		this.data = matrix;
	}
	
	
	/**
	 * Constructor with matrices.
	 * @param data matrices.
	 */
	public ImageMatrix(double[][][] data) {
		assert (data != null && data.length > 0);
		if (data.length == 1) {
			this.data = MatrixReal.Wrap(data[0]);
		}
		else {
			Matrix[] matrices = new Matrix[data.length];
			for (int d = 0; d < data.length; d++) matrices[d] = MatrixReal.Wrap(data[d]);
			this.data = new MatrixStack(matrices);
		}
	}
	
	
	/**
	 * Constructor with size and hint value. 
	 * @param size image size.
	 * @param hint hinting value.
	 */
	public ImageMatrix(Size size, NeuronValue hint) {
		this.data = MatrixUtil.create(size, hint);
	}
	
	
	/**
	 * Constructor with size and neuron channel.
	 * @param size image size.
	 * @param neuronChannel neuron channel.
	 */
	public ImageMatrix(Size size, int neuronChannel) {
		this(size, NeuronValueCreator.newNeuronValue(neuronChannel));
	}
	
	
	/**
	 * Constructor with size and neuron value creator.
	 * @param size image size.
	 * @param creator creator.
	 */
	public ImageMatrix(Size size, NeuronValueCreator creator) {
		this(size, creator.newNeuronValue());
	}
	
	
	/**
	 * Creating new matrix image.
	 * @param size size.
	 * @return new matrix image.
	 */
	public ImageMatrix create(Size size) {
		return new ImageMatrix(this.data.create(size));
	}
	
	
	/**
	 * Creating new matrix image.
	 * @return new matrix image.
	 */
	public ImageMatrix create() {
		return new ImageMatrix(this.data.create());
	}

	
	@Override
	public int getWidth() {return data.columns();}

	
	@Override
	public int getHeight() {return data.rows();}

	
	/**
	 * Getting depth.
	 * @return depth.
	 */
	public int getDepth() {return MatrixUtil.depth(data);}
	
	
	/**
	 * Getting size.
	 * @return size.
	 */
	public Size getSize() {
		return new Size(getWidth(), getHeight(), getDepth());
	}
	
	
	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 */
	public int getNeuronChannel() {
		return MatrixUtil.split(data)[0].get(0, 0).length();
	}
	
	
	@Override
	public int getLength() {return MatrixUtil.capacity(data);}


	/**
	 * Getting internal matrix data.
	 * @return matrix data.
	 */
	public Matrix get() {return data;}
	
	
	/**
	 * Getting internal matrix data at specified index.
	 * @param index specified index.
	 * @return internal matrix data at specified index.
	 */
	public Matrix get(int index) {
		return MatrixUtil.split(this.data)[index];
	}
	
	
	/**
	 * Getting internal matrix data at specified index..
	 * @param index specified index.
	 * @return matrix data at specified index.
	 */
	public double[][] getReal(int index) {
		Matrix matrix = MatrixUtil.split(this.data)[index];
		return matrix instanceof MatrixReal ? ((MatrixReal)matrix).getData() : null;
	}
	
	
	/**
	 * Getting internal matrix data.
	 * @return internal matrix data.
	 */
	public double[][][] getReals() {
		Matrix[] matrices = MatrixUtil.split(this.data);
		double[][][] arrays = new double[matrices.length][][];
		for (int d = 0; d < arrays.length; d++) {
			double[][] array = matrices[d] instanceof MatrixReal ? ((MatrixReal)matrices[d]).getData() : null;
			if (array == null) return null;
			arrays[d] = array;
		}
		return arrays;
	}
	
	
	/**
	 * Getting image.
	 * @return image.
	 */
	public BufferedImage getImage() {
		if (getNeuronChannel() != 1) throw new IllegalArgumentException();
		
		Matrix[] thisData = MatrixUtil.split(this.data);
		RasterType rasterType = Raster.toRasterType(thisData.length);
		int sourceImageType = rasterType == RasterType.GRAY ? BufferedImage.TYPE_BYTE_GRAY : SOURCE_IMAGE_TYPE_DEFAULT;
		BufferedImage image = new BufferedImage(getWidth(), getHeight(), sourceImageType);
		
		double factor = 255;
		for (int y = 0; y < getHeight(); y++) {
			for (int x = 0; x < getWidth(); x++) {
				int a = Image.ALPHA_DEFAULT, r = 0, g = 0, b = 0, gray = 0;
				
                switch (rasterType) {
                case GRAY:
	                {
						NeuronValue1 value0 = (NeuronValue1)thisData[0].get(y, x);
						gray = (int)(value0.get()*factor + 0.5);
						r = g = b = gray;
	                }
                	break;
                case GB:
	                {
						NeuronValue1 value0 = (NeuronValue1)thisData[0].get(y, x);
						NeuronValue1 value1 = (NeuronValue1)thisData[1].get(y, x);
	                	g = (int)(value0.get()*factor + 0.5);
	                	b = (int)(value1.get()*factor + 0.5);
	                }
                	break;
                case RGB:
	                {
						NeuronValue1 value0 = (NeuronValue1)thisData[0].get(y, x);
						NeuronValue1 value1 = (NeuronValue1)thisData[1].get(y, x);
						NeuronValue1 value2 = (NeuronValue1)thisData[2].get(y, x);
	                	r = (int)(value0.get()*factor + 0.5);
	                	g = (int)(value1.get()*factor + 0.5);
	                	b = (int)(value2.get()*factor + 0.5);
	                }
                	break;
                case ARGB:
	                {
						NeuronValue1 value0 = (NeuronValue1)thisData[0].get(y, x);
						NeuronValue1 value1 = (NeuronValue1)thisData[1].get(y, x);
						NeuronValue1 value2 = (NeuronValue1)thisData[2].get(y, x);
						NeuronValue1 value3 = (NeuronValue1)thisData[3].get(y, x);
	                	a = (int)(value0.get()*factor + 0.5);
	                	r = (int)(value1.get()*factor + 0.5);
	                	g = (int)(value2.get()*factor + 0.5);
	                	b = (int)(value3.get()*factor + 0.5);
	                }
                	break;
                default:
	                {
						NeuronValue1 value0 = (NeuronValue1)thisData[0].get(y, x);
						gray = (int)(value0.get()*factor + 0.5);
						r = g = b = gray;
	                }
                	break;
                }
				
				int p = (a << 24) | (r << 16) | (g << 8) | b;
	            image.setRGB(x, y, p);
	            
			} //End for x
			
		} //End for y
		
		return image;
	}
	
	
	/**
	 * Getting image.
	 * @param index index.
	 * @return image.
	 */
	public BufferedImage getImage(int index) {
		RasterType rasterType = Raster.toRasterType(getNeuronChannel());
		int sourceImageType = rasterType == RasterType.GRAY ? BufferedImage.TYPE_BYTE_GRAY : SOURCE_IMAGE_TYPE_DEFAULT;
		BufferedImage image = new BufferedImage(getWidth(), getHeight(), sourceImageType);
		
		double factor = 255;
		Matrix thisData = MatrixUtil.split(this.data)[index];
		for (int y = 0; y < getHeight(); y++) {
			for (int x = 0; x < getWidth(); x++) {
				int a = Image.ALPHA_DEFAULT, r = 0, g = 0, b = 0, gray = 0;
				
				NeuronValue value = thisData.get(y, x);
                switch (rasterType) {
                case GRAY:
					NeuronValue1 value1 = (NeuronValue1)value;
					gray = (int)(value1.get()*factor + 0.5);
					r = g = b = gray;
                	break;
                case GB:
					NeuronValueV value2 = (NeuronValueV)value;
                	g = (int)(value2.get(0)*factor + 0.5);
                	b = (int)(value2.get(1)*factor + 0.5);
                	break;
                case RGB:
					NeuronValueV value3 = (NeuronValueV)value;
                	r = (int)(value3.get(0)*factor + 0.5);
                	g = (int)(value3.get(1)*factor + 0.5);
                	b = (int)(value3.get(2)*factor + 0.5);
                	break;
                case ARGB:
					NeuronValueV value4 = (NeuronValueV)value;
                	a = (int)(value4.get(0)*factor + 0.5);
                	r = (int)(value4.get(1)*factor + 0.5);
                	g = (int)(value4.get(2)*factor + 0.5);
                	b = (int)(value4.get(3)*factor + 0.5);
                	break;
                default:
					NeuronValue1 d = (NeuronValue1)value;
					gray = (int)(d.get()*factor + 0.5);
					r = g = b = gray;
                	break;
                }
				
				int p = (a << 24) | (r << 16) | (g << 8) | b;
	            image.setRGB(x, y, p);
	            
			} //End for x
			
		} //End for y
		
		return image;
	}
	
	
	/**
	 * Getting images.
	 * @return images.
	 */
	public List<BufferedImage> getImages() {
		List<BufferedImage> images = Util.newList(0);
		int depth = getDepth();
		for (int d = 0; d < depth; d++) {
			BufferedImage image = getImage(d);
			assert (image != null);
			images.add(image);
		}
		return images;
	}
	
	
	/**
	 * Getting image list.
	 * @return image list.
	 */
	public ImageList getImageList() {
		Matrix[] matrices = MatrixUtil.split(this.data);
		List<Image> images = Util.newList(matrices.length);
		for (Matrix matrix : matrices) images.add(new ImageMatrix(matrix));
		return ImageList.create(images);
	}
	
	
	/**
	 * Converting this image to raster.
	 * @return raster.
	 */
	public Raster toRaster() {
		if (getDepth() > 1) {
			if (getNeuronChannel() == 1) {
				return Raster2DImpl.create(this);
			}
			else {
				ImageList imageList = getImageList();
				return Raster3DImpl.create(imageList);
			}
		}
		else if (getHeight() > 1)
			return Raster2DImpl.create(this);
		else
			return Raster1DImpl.create(this);
	}
	
	
	@Override
	public boolean save(Path path) {
		return toRaster().save(path);
	}

	
	@Override
	public NeuronValue[] convertFromSoundToNeuronValues(int neuronChannel, int length, boolean isNorm) {
		assert (getNeuronChannel() == neuronChannel && getLength() == length && isNorm);
		return MatrixUtil.extractValues(data);
	}

	
	@Override
	public NeuronValue[] convertFromImageToNeuronValues(int neuronChannel, int width, int height, boolean isNorm) {
		assert (getNeuronChannel() == neuronChannel && this.data.columns() == width && this.data.rows() == height && isNorm);
		return MatrixUtil.extractValues(data);
	}


}
