/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.awt.Dimension;
import java.nio.file.Paths;
import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.conv.ConvLayer;
import net.ea.ann.conv.ConvNetworkAbstract;
import net.ea.ann.conv.ConvNetworkImpl;
import net.ea.ann.conv.Filter;
import net.ea.ann.conv.FilterFactory;
import net.ea.ann.conv.FilterFactoryImpl;
import net.ea.ann.conv.ImageSpec;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
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
	 * Encoding convolutional network.
	 */
	protected ConvNetworkAbstract encodeConv = null;

	
	/**
	 * Decoding convolutional network.
	 */
	protected ConvNetworkAbstract decodeConv = null;
	
	
	/**
	 * Constructor with neuron channel, width, height, and identifier reference.
	 * @param neuronChannel image type.
	 * @param width raster width.
	 * @param height raster height.
	 * @param idRef identifier reference.
	 */
	public ConvVAEImpl(int neuronChannel, int width, int height, Id idRef) {
		super(neuronChannel, null, idRef);
		this.activateRef = ImageSpec.toActivationRef(neuronChannel, isNorm());
		
		this.config.put(LEARN_MAX_ITERATION_FIELD, 1);

		this.config.put(ImageSpec.SOURCE_IMAGE_TYPE_FIELD, ImageSpec.SOURCE_IMAGE_TYPE_DEFAULT);
		this.config.put(ImageSpec.NORM_FIELD, ImageSpec.NORM_DEFAULT);
		this.config.put(ImageSpec.SOURCE_IMAGE_RESIZE_FIELD, ImageSpec.SOURCE_IMAGE_RESIZE_DEFAULT);
		this.config.put(ImageSpec.ALPHA_FIELD, ImageSpec.ALPHA_DEFAULT);
		
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
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param encodeFilters encoding filters.
	 * @param decodeFilters decoding filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode,
			Filter[] encodeFilters, Filter[] decodeFilters) {
//		reset();
		int xDim = 0;
		
		if (encodeFilters != null && encodeFilters.length > 0) {
			encodeConv = createEncodeConvNetwork();
			if (encodeConv == null)
				return false;
			else if (!encodeConv.initialize(width, height, encodeFilters))
				return false;
			
			xDim = encodeConv.getConvOutputLayer().getWidth() * encodeConv.getConvOutputLayer().getHeight();
		}
		else
			xDim = width * height;

		if(!super.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode))
			return false;

		if (decodeFilters != null && decodeFilters.length > 0) {
			//Fixing here.
		}
		
		return true;
	}
	
	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param encodeFilters encoding filters.
	 * @param decodeFilters decoding filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, 
			Filter[] encodeFilters, Filter[] decodeFilters) {
		return this.initialize(zDim, nHiddenNeuronEncode, null, encodeFilters, decodeFilters);
	}

	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param encodeFilters encoding filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, 
			Filter[] encodeFilters) {
		return this.initialize(zDim, nHiddenNeuronEncode, null, encodeFilters, null);
	}

	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode) {
		return this.initialize(zDim, nHiddenNeuronEncode, null, null, null);
	}

	
	/**
	 * Creating encoding convolutional neural network.
	 * @return encoding convolutional neural network.
	 */
	protected ConvNetworkAbstract createEncodeConvNetwork() {
		return ConvNetworkImpl.create(neuronChannel, idRef);
	}
	
	
	/**
	 * Getting encoding filter factory.
	 * @return encoding filter factory.
	 */
	protected FilterFactory getEncodeFilterFactory() {
		ConvNetworkAbstract conv = createEncodeConvNetwork();
		if (conv == null) return null;
		
		ConvLayer layer = conv.newLayer(1, 1, null);
		if (layer == null)
			return null;
		else
			return new FilterFactoryImpl(layer);
	}
	
	
	@Override
	public synchronized NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		for (Record record : sample) {
			try {
				if ((record.undefinedInput == null) || !(record.undefinedInput instanceof ImageSpec)) continue;
				
				ImageSpec imageSpec = (ImageSpec)record.undefinedInput;
				if (imageSpec == null) continue;
				
				NeuronValue[] input = ImageSpec.convertFromImageToNeuronValues(neuronChannel, width, height,
						imageSpec, getSourceImageType(), isSourceImageResize(), isNorm());
				if (input != null) record.input = input;
			}
			catch (Throwable e) {
				Util.trace(e);
			}
		}
		
		return super.learn(sample);
	}


	@Override
	public synchronized NeuronValue[] learnByImages(Iterable<ImageSpec> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return bpLearnByImages(sample, learningRate, terminatedThreshold, maxIteration);
	}


	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected NeuronValue[] bpLearnByImages(Iterable<ImageSpec> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			for (ImageSpec imageSpec : sample) {
				if (imageSpec == null) continue;
				
				NeuronValue[] input = null;
				
				//Evaluating convolutional encoding network.
				if (encodeConv != null) {
					try {
						input = encodeConv.evaluateByImage(imageSpec);
					} catch (Throwable e) {Util.trace(e);}
				}
				else
					input = ImageSpec.convertFromImageToNeuronValues(neuronChannel, width, height,
						imageSpec, getSourceImageType(), isSourceImageResize(), isNorm());
				
				if (input == null) continue;
					
				//Evaluating encoder.
				try {
					Record encodeRecord = new Record();
					encodeRecord.input = input;
					encoder.evaluate(encodeRecord, true);
				} catch (Throwable e) {Util.trace(e);}
				
				//Evaluating decoder.
				try {
					Record decodeRecord = new Record();
					decodeRecord.input = randomizeDataZ(learnRnd);
					decodeRecord.output = input;
					decoder.evaluate(decodeRecord, true);
				} catch (Throwable e) {Util.trace(e);}

				
				try {
					//Updating weights and biases of encoder.
					List<LayerStandard> encoderBackbone = encoder.getBackbone();
					encoder.bpLearn(encoderBackbone, input, null, learningRate, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
				
				try {
					//Updating weights and biases of encoder.
					List<LayerStandard> decoderBackbone = decoder.getBackbone();
					error = decoder.bpLearn(decoderBackbone, encoder.getOutputLayer().getOutput(), input, learningRate, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "vae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0)
				doStarted = false;
			else {
				double errorMean = 0;
				for (NeuronValue r : error) errorMean += r.norm();
				errorMean = errorMean / error.length;
				if (errorMean < terminatedThreshold) doStarted = false; 
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "vae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	@Override
	public synchronized ImageSpec generateImage() throws RemoteException {
		try {
			NeuronValue[] dataX = generate();
			Dimension size = getEncodeRasterSize();
			return ImageSpec.convertFromNeuronValuesToImage(dataX, neuronChannel, size.width, size.height,
				getSourceImageType(), isNorm(), getDefaultAlpha());
		} catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;
	}
	
	
	/**
	 * Getting encoding raster size.
	 * @return encoding raster size.
	 */
	protected Dimension getEncodeRasterSize() {
		if (encodeConv == null) return new Dimension(width, height);
		
		ConvLayer convOutputLayer = encodeConv.getConvOutputLayer();
		if (convOutputLayer == null)
			return new Dimension(width, height);
		else
			return new Dimension(convOutputLayer.getWidth(), convOutputLayer.getHeight());
		
	}
	
	
	/**
	 * Getting source image type.
	 * @return source image type.
	 */
	private int getSourceImageType() {
		if (config.containsKey(ImageSpec.SOURCE_IMAGE_TYPE_FIELD))
			return config.getAsInt(ImageSpec.SOURCE_IMAGE_TYPE_FIELD);
		else
			return ImageSpec.SOURCE_IMAGE_TYPE_DEFAULT;
	}
	
	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	private boolean isNorm() {
		if (config.containsKey(ImageSpec.NORM_FIELD))
			return config.getAsBoolean(ImageSpec.NORM_FIELD);
		else
			return ImageSpec.NORM_DEFAULT;
	}
	
	
	/**
	 * Getting whether source image is resized.
	 * @return whether source image is resized.
	 */
	private boolean isSourceImageResize() {
		if (config.containsKey(ImageSpec.SOURCE_IMAGE_RESIZE_FIELD))
			return config.getAsBoolean(ImageSpec.SOURCE_IMAGE_RESIZE_FIELD);
		else
			return ImageSpec.SOURCE_IMAGE_RESIZE_DEFAULT;
	}
	
	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	private int getDefaultAlpha() {
		if (config.containsKey(ImageSpec.ALPHA_FIELD))
			return config.getAsInt(ImageSpec.ALPHA_FIELD);
		else
			return ImageSpec.ALPHA_DEFAULT;
	}
	

	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		try (ConvVAEImpl convVAE = new ConvVAEImpl(3, 250, 250)) {
			convVAE.config.put(LEARN_MAX_ITERATION_FIELD, 1);
			
			Filter[] filters = new Filter[] {convVAE.getEncodeFilterFactory().meanFilter(3, 3)};
			//convVAE.initialize(10, new int[] {30, 20});
			convVAE.initialize(10, new int[] {30, 20}, filters);
		
			List<ImageSpec> sample = Util.newList(0);
			
			ImageSpec image = ImageSpec.load(Paths.get("working/sample1.png"));
			sample.add(image);
			
			image = ImageSpec.load(Paths.get("working/sample2.png"));
			sample.add(image);

			image = ImageSpec.load(Paths.get("working/sample3.png"));
			sample.add(image);

			convVAE.learnByImages(sample);
			
			image = convVAE.generateImage();
			image.save(Paths.get("working/gen1.png"));
			
			image = convVAE.generateImage();
			image.save(Paths.get("working/gen2.png"));
			
			image = convVAE.generateImage();
			image.save(Paths.get("working/gen3.png"));
		}
		catch (Exception e) {
			Util.trace(e);
		}
	}


}
