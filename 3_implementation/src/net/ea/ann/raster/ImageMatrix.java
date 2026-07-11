/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.image.BufferedImage;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;

import javax.imageio.ImageIO;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
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
		this.data = data;
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
	 * Getting neuron channel.
	 * @return neuron channel.
	 */
	public int getNeuronChannel() {return data.get(0, 0).length();}
	
	
	@Override
	public int getLength() {return MatrixUtil.capacity(data);}


	/**
	 * Getting internal matrix data.
	 * @return matrix data.
	 */
	public Matrix get() {return data;}
	
	
	/**
	 * Getting image.
	 * @return image.
	 */
	public BufferedImage getImage() {return getImage(0);}
	
	
	/**
	 * Getting image.
	 * @param index index.
	 * @return image.
	 */
	BufferedImage getImage(int index) {
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
	 * Converting this image to raster.
	 * @return raster.
	 */
	public Raster toRaster() {
		if (getDepth() > 1) {
			Matrix[] matrices = MatrixUtil.split(this.data);
			List<Image> images = Util.newList(matrices.length);
			for (Matrix matrix : matrices) images.add(new ImageMatrix(matrix));
			ImageList imageList = ImageList.create(images);
			return Raster3DImpl.create(imageList);
		}
		else if (getHeight() > 1)
			return Raster2DImpl.create(this);
		else
			return Raster1DImpl.create(this);
	}
	
	
	@Override
	public boolean save(Path path) {
		try {
			BufferedImage image = getImage();
			if (image == null) return false;
			
			OutputStream os = Files.newOutputStream(path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
			ImageIO.write(image, Image.IMAGE_FORMAT_DEFAULT, os);
			os.close();
			
			return true;
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return false;
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
