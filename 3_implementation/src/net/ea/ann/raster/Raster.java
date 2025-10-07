/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.io.Serializable;
import java.nio.file.Path;

import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.Logistic1;
import net.ea.ann.core.function.LogisticV;
import net.ea.ann.core.function.ReLU1;
import net.ea.ann.core.function.ReLUV;
import net.ea.ann.core.function.indexed.IndexedLogistic1;
import net.ea.ann.core.function.indexed.IndexedLogisticV;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represent an raster.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Raster extends Serializable, Cloneable {

	
	/**
	 * Name of flag to normalize pixel in rang [0, 1].
	 */
	String NORM_FIELD = "raster_norm";

	
	/**
	 * Flag to normalize raster point in range [0, 1].
	 */
	boolean NORM_DEFAULT = true;

	
//	/**
//	 * Name of resizing source flag.
//	 */
//	static final String SOURCE_RESIZE_FIELD = "raster_resize";

	
//	/**
//	 * Resizing source flag.
//	 */
//	boolean SOURCE_RESIZE_DEFAULT = true;
	
	
	/**
	 * This enum represents raster type. This enum is general for digital data such as sound, image, video.
	 * @author Loc Nguyen
	 *
	 */
	enum RasterType {
		
		/**
		 * Single channel.
		 */
		GRAY,
		
		/**
		 * Pair channel
		 */
		GB,

		/**
		 * Triple channel
		 */
		RGB,
		
		/**
		 * Quadruplet channel
		 */
		ARGB,
		
	}
	

	/**
	 * Getting ID of raster.
	 * @return ID of raster.
	 */
	int id();
	
	
	/**
	 * Getting raster width.
	 * @return raster width.
	 */
	int getWidth();
	
	
	/**
	 * Getting raster height.
	 * @return raster height.
	 */
	int getHeight();
	

	/**
	 * Getting raster depth.
	 * @return raster depth.
	 */
	int getDepth();
	

	/**
	 * Getting raster time.
	 * @return raster time.
	 */
	int getTime();

	
	/**
	 * Getting representative image. An 3D raster like video should also have representative image.
	 * @return representative image.
	 */
	java.awt.Image getRepImage();
	
	
	/**
	 * Getting default format.
	 * @return default format.
	 */
	String getDefaultFormat();
	
	
	/**
	 * Getting property.
	 * @return property.
	 */
	RasterProperty getProperty();
	
	
	/**
	 * Save raster to path.
	 * @param path specified path.
	 * @return true if writing is successful.
	 */
	boolean save(Path path);
	
	
	/**
	 * Extracting image into neuron value array.
	 * @param layer specified layer.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return neuron value array.
	 */
	NeuronValue[] toNeuronValues(ConvLayerSingle layer, boolean isNorm);

		
	/**
	 * Extracting raster into neuron value array.
	 * @param neuronChannel neuron channel.
	 * @param size raster size.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return neuron value array.
	 */
	NeuronValue[] toNeuronValues(int neuronChannel, Size size, boolean isNorm);
	
	
	/**
	 * Convert neuron channel to raster type.
	 * @param neuronChannel neuron channel.
	 * @return raster type.
	 */
	static RasterType toRasterType(int neuronChannel) {
		RasterType rasterType = RasterType.GRAY;
        switch (neuronChannel) {
        case 1:
        	rasterType = RasterType.GRAY;
        	break;
        case 2:
        	rasterType = RasterType.GB;
        	break;
        case 3:
        	rasterType = RasterType.RGB;
        	break;
        case 4:
        	rasterType = RasterType.ARGB;
        	break;
        default:
        	rasterType = RasterType.GRAY;
        	break;
        }
        
        return rasterType;
	}
	
	
	/**
	 * Retrieving activation function from raster type. 
	 * @param rasterType raster type.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return activation function from raster type.
	 */
	private static Function toActivationRef(RasterType rasterType, boolean isNorm) {
		Function f = null;
		
        switch (rasterType) {
        case GRAY:
        	f = isNorm ? new Logistic1(0.0, 1.0) : new Logistic1(0.0, 255.0);
        	break;
        case GB:
        	f = isNorm ? new LogisticV(2, 0.0, 1.0) : new LogisticV(2, 0.0, 255.0);
        	break;
        case RGB:
        	f = isNorm ? new LogisticV(3, 0.0, 1.0) : new LogisticV(3, 0.0, 255.0);
        	break;
        case ARGB:
        	f = isNorm ? new LogisticV(4, 0.0, 1.0) : new LogisticV(4, 0.0, 255.0);
        	break;
        default:
        	f = isNorm ? new Logistic1(0.0, 1.0) : new Logistic1(0.0, 255.0);
        	break;
        }
        
        return f;
	}


	/**
	 * Retrieving indexed activation function from raster type. 
	 * @param rasterType raster type.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return activation function from raster type.
	 */
	private static Function toActivationRefIndexed(RasterType rasterType, boolean isNorm) {
		Function f = null;
		
        switch (rasterType) {
        case GRAY:
        	f = isNorm ? new IndexedLogistic1(0.0, 1.0) : new IndexedLogistic1(0.0, 255.0);
        	break;
        case GB:
        	f = isNorm ? new IndexedLogisticV(2, 0.0, 1.0) : new IndexedLogisticV(2, 0.0, 255.0);
        	break;
        case RGB:
        	f = isNorm ? new IndexedLogisticV(3, 0.0, 1.0) : new IndexedLogisticV(3, 0.0, 255.0);
        	break;
        case ARGB:
        	f = isNorm ? new IndexedLogisticV(4, 0.0, 1.0) : new IndexedLogisticV(4, 0.0, 255.0);
        	break;
        default:
        	f = isNorm ? new IndexedLogistic1(0.0, 1.0) : new IndexedLogistic1(0.0, 255.0);
        	break;
        }
        
        return f;
	}

	
	/**
	 * Retrieving convolutional activation function from raster type. 
	 * @param rasterType raster type.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return convolutional activation function from raster type.
	 */
	private static Function toConvActivationRef(RasterType rasterType, boolean isNorm) {
		Function f = null;
		
        switch (rasterType) {
        case GRAY:
        	f = isNorm ? new ReLU1(0.0, 1.0) : new ReLU1(0.0, 255.0);
        	break;
        case GB:
        	f = isNorm ? new ReLUV(2, 0.0, 1.0) : new ReLUV(2, 0.0, 255.0);
        	break;
        case RGB:
        	f = isNorm ? new ReLUV(3, 0.0, 1.0) : new ReLUV(3, 0.0, 255.0);
        	break;
        case ARGB:
        	f = isNorm ? new ReLUV(4, 0.0, 1.0) : new ReLUV(4, 0.0, 255.0);
        	break;
        default:
        	f = isNorm ? new ReLU1(0.0, 1.0) : new ReLU1(0.0, 255.0);
        	break;
        }
        
        return f;
	}

	
	/**
	 * Retrieving activation function from neuron channel.
	 * @param neuronChannel neuron channel.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return activation function from neuron channel.
	 */
	static Function toActivationRef(int neuronChannel, boolean isNorm) {
		if (isNorm) {
			if (neuronChannel <= 0)
				return null;
			else if (neuronChannel == 1)
	        	return new Logistic1(0.0, 1.0);
			else
	        	return new LogisticV(neuronChannel, 0.0, 1.0);
		}
		
		RasterType rasterType = toRasterType(neuronChannel);
		return toActivationRef(rasterType, isNorm);
	}

	
	/**
	 * Retrieving indexed activation function from neuron channel.
	 * @param neuronChannel neuron channel.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return indexed activation function from neuron channel.
	 */
	static Function toActivationRefIndexed(int neuronChannel, boolean isNorm) {
		if (isNorm) {
			if (neuronChannel <= 0)
				return null;
			else if (neuronChannel == 1)
	        	return new IndexedLogistic1(0.0, 1.0);
			else
	        	return new IndexedLogisticV(neuronChannel, 0.0, 1.0);
		}
		
		RasterType rasterType = toRasterType(neuronChannel);
		return toActivationRefIndexed(rasterType, isNorm);
	}

	
	/**
	 * Retrieving activation function from neuron channel.
	 * @param neuronChannel neuron channel.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return activation function from neuron channel.
	 */
	static Function toConvActivationRef(int neuronChannel, boolean isNorm) {
		if (isNorm) {
			if (neuronChannel <= 0)
				return null;
			else if (neuronChannel == 1)
	        	return new ReLU1(0.0, 1.0);
			else
	        	return new ReLUV(neuronChannel, 0.0, 1.0);
		}
		
		RasterType rasterType = toRasterType(neuronChannel);
		return toConvActivationRef(rasterType, isNorm);
	}

	
	/**
	 * Retrieving rectified linear unit (ReLU) activation function from neuron channel.
	 * @param neuronChannel neuron channel.
	 * @param isNorm checking whether to normalized pixels value in range [0, 1].
	 * @return rectified linear unit (ReLU) activation function from neuron channel.
	 */
	static Function toReLUActivationRef(int neuronChannel, boolean isNorm) {
		return toConvActivationRef(neuronChannel, isNorm);
	}


}
