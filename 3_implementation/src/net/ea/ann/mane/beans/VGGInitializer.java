/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.awt.Dimension;
import java.io.Serializable;

import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class provides initialization methods for VGG model.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class VGGInitializer implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal VGG.
	 */
	protected VGG vgg = null;
	
	
	/**
	 * Constructor with VGG model.
	 * @param vgg VGG model.
	 */
	public VGGInitializer(VGG vgg) {
		this.vgg = vgg;
	}

	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param inputDepth input depth.
	 * @param middleSize middle size.
	 * @param middleDepth middle depth.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, int inputDepth, Dimension middleSize, int middleDepth, Dimension outputSize, int outputDepth) {
		inputDepth = inputDepth < 1 ? 1 : inputDepth; 
		middleDepth = middleDepth < 1 ? 1 : middleDepth; 
		outputDepth = outputDepth < 1 ? 1 : outputDepth;
		return vgg.initialize(
			new Size(inputSize.width, inputSize.height, inputDepth, 1),
			new Size(middleSize.width, middleSize.height, middleDepth, 1),
			new Size(outputSize.width, outputSize.height, outputDepth, 1));
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param outputSize output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Dimension middleSize, Dimension outputSize) {
		return initialize(inputSize, 1, middleSize, 1, outputSize, 1);
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param inputDepth input depth.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, int inputDepth, Dimension outputSize, int outputDepth) {
		inputDepth = inputDepth < 1 ? 1 : inputDepth; 
		outputDepth = outputDepth < 1 ? 1 : outputDepth;
		return vgg.initialize(
			new Size(inputSize.width, inputSize.height, inputDepth, 1),
			new Size(outputSize.width, outputSize.height, outputDepth, 1));
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param outputSize output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Dimension outputSize) {
		return initialize(inputSize, 1, outputSize, 1);
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param inputDepth input depth.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if initialization is successful.
	 */
	public boolean initializeWithImplicitMiddleSize(Dimension inputSize, int inputDepth, Dimension outputSize, int outputDepth) {
		inputDepth = inputDepth < 1 ? 1 : inputDepth; 
		outputDepth = outputDepth < 1 ? 1 : outputDepth;
		return vgg.initializeWithImplicitMiddleSize(
			new Size(inputSize.width, inputSize.height, inputDepth, 1),
			new Size(outputSize.width, outputSize.height, outputDepth, 1));
	}

	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param outputSize output size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeWithImplicitMiddleSize(Dimension inputSize, Dimension outputSize) {
		return initializeWithImplicitMiddleSize(inputSize, 1, outputSize, 1);
	}

	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @param middleDepth middle depth.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @param nCoreClasses core classes size.
	 * @param nCoreClassesDepth core classes depth.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, Dimension middleSize, int middleDepth, Dimension outputSize, int outputDepth, Dimension nCoreClasses, int nCoreClassesDepth) {
		if (!(vgg instanceof VGGExt)) return false;
		middleDepth = middleDepth < 1 ? 1 : middleDepth; 
		outputDepth = outputDepth < 1 ? 1 : outputDepth;
		nCoreClassesDepth = nCoreClassesDepth < 1 ? 1 : nCoreClassesDepth;
		VGGExt extVGG = (VGGExt)vgg;
		return extVGG.initialize(sample,
			new Size(middleSize.width, middleSize.height, middleDepth, 1),
			new Size(outputSize.width, outputSize.height, outputDepth, 1),
			new Size(nCoreClasses.width, nCoreClasses.height, nCoreClassesDepth, 1));
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @param outputSize output size.
	 * @param nCoreClasses core classes size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, Dimension middleSize, Dimension outputSize, Dimension nCoreClasses) {
		return initialize(sample, middleSize, 1, outputSize, 1, nCoreClasses, 1);
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @param middleDepth middle depth.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, Dimension middleSize, int middleDepth, Dimension outputSize, int outputDepth) {
		if (!(vgg instanceof VGGExt)) return false;
		middleDepth = middleDepth < 1 ? 1 : middleDepth; 
		outputDepth = outputDepth < 1 ? 1 : outputDepth;
		VGGExt extVGG = (VGGExt)vgg;
		return extVGG.initialize(sample,
			new Size(middleSize.width, middleSize.height, middleDepth, 1),
			new Size(outputSize.width, outputSize.height, outputDepth, 1));
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @param outputSize output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, Dimension middleSize, Dimension outputSize) {
		return initialize(sample, middleSize, 1, outputSize, 1);
	}

	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, Dimension outputSize, int outputDepth) {
		if (!(vgg instanceof VGGExt)) return false;
		outputDepth = outputDepth < 1 ? 1 : outputDepth;
		VGGExt extVGG = (VGGExt)vgg;
		return extVGG.initialize(sample,
			new Size(outputSize.width, outputSize.height, outputDepth, 1));
	}


	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param outputSize output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, Dimension outputSize) {return initialize(sample, outputSize, 1);}


	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if initialization is successful.
	 */
	public boolean initializeWithImplicitMiddleSize(Iterable<Raster> sample, Dimension outputSize, int outputDepth) {
		if (!(vgg instanceof VGGExt)) return false;
		outputDepth = outputDepth < 1 ? 1 : outputDepth;
		VGGExt extVGG = (VGGExt)vgg;
		return extVGG.initializeWithImplicitMiddleSize(sample,
			new Size(outputSize.width, outputSize.height, outputDepth, 1));
	}


	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param outputSize output size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeWithImplicitMiddleSize(Iterable<Raster> sample, Dimension outputSize) {
		return initializeWithImplicitMiddleSize(sample, outputSize, 1);
	}
	
	
}
