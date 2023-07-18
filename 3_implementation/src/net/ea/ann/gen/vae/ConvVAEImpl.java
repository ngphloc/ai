/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.awt.image.BufferedImage;
import java.nio.file.Paths;
import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.conv.SerializableImage;
import net.ea.ann.core.Id;
import net.ea.ann.core.NeuronValue;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;

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
	 * Raster width.
	 */
	protected int width = 0;
	
	
	/**
	 * Raster height.
	 */
	protected int height = 0;

	
	/**
	 * Constructor with neuron channel, width, height, and identifier reference.
	 * @param neuronChannel image type.
	 * @param width raster width.
	 * @param height raster height.
	 * @param idRef identifier reference.
	 */
	public ConvVAEImpl(int neuronChannel, int width, int height, Id idRef) {
		super(neuronChannel, null, idRef);
		this.activateRef = SerializableImage.toActivationRef(neuronChannel, isNorm());
		
		this.config.put(LEARN_MAX_ITERATION_FIELD, 1);

		this.config.put(SerializableImage.SOURCE_IMAGE_TYPE_FIELD, SerializableImage.SOURCE_IMAGE_TYPE_DEFAULT);
		this.config.put(SerializableImage.NORM_FIELD, SerializableImage.NORM_DEFAULT);
		this.config.put(SerializableImage.SOURCE_IMAGE_RESIZE_FIELD, SerializableImage.SOURCE_IMAGE_RESIZE_DEFAULT);
		this.config.put(SerializableImage.ALPHA_FIELD, SerializableImage.ALPHA_DEFAULT);
		
		this.width = width;
		this.height = height;
	}

	
	/**
	 * Constructor with image type, width, and height.
	 * @param neuronChannel neuron channel.
	 * @param width raster width.
	 * @param height raster height.
	 */
	public ConvVAEImpl(int neuronChannel, int width, int height) {
		this(neuronChannel, width, height, null);
	}

	
//	/**
//	 * Resetting data structures for initialization.
//	 */
//	protected void reset() {
//		width = 0;
//		height = 0;
//	}

	
	/**
	 * Initialize with Z dimension as well as hidden neurons.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode) {
//		reset();
		int xDim = width * height;
		return super.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode);
	}
	
	
	/**
	 * Initialize with Z dimension as well as hidden neurons.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode) {
		int xDim = width * height;
		return super.initialize(xDim, zDim, nHiddenNeuronEncode);
	}

	
	@Override
	public synchronized NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		for (Record record : sample) {
			try {
				if ((record.undefinedInput == null) || !(record.undefinedInput instanceof SerializableImage)) continue;
				
				SerializableImage image = (SerializableImage)record.undefinedInput;
				if (image.getImage() == null) continue;
				
				NeuronValue[] input = SerializableImage.convertFromImageToNeuronValues(neuronChannel, width, height,
						image.getImage(), getSourceImageType(), isSourceImageResize(), isNorm());
				if (input != null) record.input = input;
			}
			catch (Throwable e) {
				Util.trace(e);
			}
		}
		
		return super.learn(sample);
	}


	@Override
	public synchronized NeuronValue[] learnByImages(Iterable<SerializableImage> sample) throws RemoteException {
		List<Record> records = Util.newList(0);
		for (SerializableImage image : sample) {
			if (image == null || image.getImage() == null) continue;
			
			Record record = new Record();
			record.undefinedInput = image;
			records.add(record);
		}
		
		return learn(records);
	}


	@Override
	public synchronized SerializableImage generateImage() throws RemoteException {
		try {
			NeuronValue[] dataX = generate();
			BufferedImage image = SerializableImage.convertFromNeuronValuesToImage(dataX, neuronChannel, width, height,
					getSourceImageType(), isNorm(), getDefaultAlpha());
			if (image == null)
				return null;
			else
				return new SerializableImage(image);
			
		} catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;
	}
	
	
	/**
	 * Getting source image type.
	 * @return source image type.
	 */
	private int getSourceImageType() {
		if (config.containsKey(SerializableImage.SOURCE_IMAGE_TYPE_FIELD))
			return config.getAsInt(SerializableImage.SOURCE_IMAGE_TYPE_FIELD);
		else
			return SerializableImage.SOURCE_IMAGE_TYPE_DEFAULT;
	}
	
	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	private boolean isNorm() {
		if (config.containsKey(SerializableImage.NORM_FIELD))
			return config.getAsBoolean(SerializableImage.NORM_FIELD);
		else
			return SerializableImage.NORM_DEFAULT;
	}
	
	
	/**
	 * Getting whether source image is resized.
	 * @return whether source image is resized.
	 */
	private boolean isSourceImageResize() {
		if (config.containsKey(SerializableImage.SOURCE_IMAGE_RESIZE_FIELD))
			return config.getAsBoolean(SerializableImage.SOURCE_IMAGE_RESIZE_FIELD);
		else
			return SerializableImage.SOURCE_IMAGE_RESIZE_DEFAULT;
	}
	
	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	private int getDefaultAlpha() {
		if (config.containsKey(SerializableImage.ALPHA_FIELD))
			return config.getAsInt(SerializableImage.ALPHA_FIELD);
		else
			return SerializableImage.ALPHA_DEFAULT;
	}
	

	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		try (ConvVAEImpl convVAE = new ConvVAEImpl(3, 40, 40)) {
			convVAE.config.put(LEARN_MAX_ITERATION_FIELD, 20);
			convVAE.initialize(10, new int[] {30, 20});
		
			List<Record> sample = Util.newList(0);
			
			Record record = new Record();
			record.undefinedInput = SerializableImage.load(Paths.get("working/sample3.png"));
			sample.add(record);
			
			record = new Record();
			record.undefinedInput = SerializableImage.load(Paths.get("working/sample2.png"));
			sample.add(record);

			record = new Record();
			record.undefinedInput = SerializableImage.load(Paths.get("working/sample1.png"));
			sample.add(record);

			convVAE.learn(sample);
			
			SerializableImage image = convVAE.generateImage();
			image.save(Paths.get("working/gen.png"));
		}
		catch (Exception e) {
			Util.trace(e);
		}
	}


}
