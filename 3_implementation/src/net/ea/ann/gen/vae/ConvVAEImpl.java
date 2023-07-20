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
import net.ea.ann.conv.DeconvNetworkImpl;
import net.ea.ann.conv.Raster;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.conv.filter.FilterFactoryImpl;
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
	 * Convolutional network.
	 */
	protected ConvNetworkAbstract conv = null;

	
	/**
	 * Deconvolutional network.
	 */
	protected ConvNetworkAbstract deconv = null;
	
	
	/**
	 * Constructor with neuron channel, width, height, and identifier reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param width raster width.
	 * @param height raster height.
	 * @param idRef identifier reference.
	 */
	public ConvVAEImpl(int neuronChannel, int width, int height, Id idRef) {
		super(neuronChannel, null, idRef);
		this.activateRef = Raster.toActivationRef(neuronChannel, isNorm());
		
		this.config.put(LEARN_MAX_ITERATION_FIELD, 1);

		this.config.put(Raster.SOURCE_IMAGE_TYPE_FIELD, Raster.SOURCE_IMAGE_TYPE_DEFAULT);
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		this.config.put(Raster.SOURCE_RESIZE_FIELD, Raster.SOURCE_RESIZE_DEFAULT);
		this.config.put(Raster.ALPHA_FIELD, Raster.ALPHA_DEFAULT);
		
		this.width = width;
		this.height = height;
	}

	
	/**
	 * Constructor with neuron channel, width, and height.
	 * @param neuronChannel neuron channel or depth.
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
	 * @param convFilters convolutional filters.
	 * @param deconvFilters deconvolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode,
			Filter[] convFilters, Filter[] deconvFilters) {
//		reset();
		int xDim = 0;
		
		if (convFilters != null && convFilters.length > 0) {
			conv = createConvNetwork();
			if (conv == null)
				return false;
			else if (!conv.initialize(width, height, convFilters))
				return false;
			
			xDim = conv.getConvOutputLayer().getWidth() * conv.getConvOutputLayer().getHeight();
		}
		else
			xDim = width * height;

		if(!super.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode))
			return false;

		if (deconvFilters != null && deconvFilters.length > 0) {
			int deconvWidth = width, deconvHeight = height;
			if (conv != null) {
				deconvWidth = conv.getConvOutputLayer().getWidth();
				deconvHeight = conv.getConvOutputLayer().getHeight();
			}
			
			deconv = createDeconvNetwork();
			if (deconv == null)
				return false;
			else if (!deconv.initialize(deconvWidth, deconvHeight, deconvFilters))
				return false;
		}
		
		return true;
	}
	
	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters deconvolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, 
			Filter[] convFilters, Filter[] deconvFilters) {
		return this.initialize(zDim, nHiddenNeuronEncode, null, convFilters, deconvFilters);
	}

	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param convFilters convolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, 
			Filter[] convFilters) {
		return this.initialize(zDim, nHiddenNeuronEncode, null, convFilters, null);
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
	 * Creating convolutional neural network.
	 * @return convolutional neural network.
	 */
	protected ConvNetworkAbstract createConvNetwork() {
		return ConvNetworkImpl.create(neuronChannel, idRef);
	}
	
	
	/**
	 * Creating deconvolutional neural network.
	 * @return deconvolutional neural network.
	 */
	protected ConvNetworkAbstract createDeconvNetwork() {
		return DeconvNetworkImpl.create(neuronChannel, idRef);
	}

	
	/**
	 * Getting filter factory.
	 * @return filter factory.
	 */
	protected FilterFactory getFilterFactory() {
		ConvNetworkAbstract conv = createConvNetwork();
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
				if ((record.undefinedInput == null) || !(record.undefinedInput instanceof Raster)) continue;
				
				Raster raster = (Raster)record.undefinedInput;
				if (raster == null) continue;
				
				NeuronValue[] input = Raster.convertFromRasterToNeuronValues(neuronChannel, width, height,
						raster, getSourceImageType(), isSourceRasterResize(), isNorm());
				if (input != null) record.input = input;
			}
			catch (Throwable e) {
				Util.trace(e);
			}
		}
		
		return super.learn(sample);
	}


	@Override
	public synchronized NeuronValue[] learnByRaster(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return bpLearnByRaster(sample, learningRate, terminatedThreshold, maxIteration);
	}


	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected NeuronValue[] bpLearnByRaster(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
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
			for (Raster raster : sample) {
				if (raster == null) continue;
				
				NeuronValue[] input = null;
				
				//Evaluating convolutional encoding network.
				if (conv != null) {
					try {
						input = conv.evaluateByRaster(raster);
					} catch (Throwable e) {Util.trace(e);}
				}
				else
					input = Raster.convertFromRasterToNeuronValues(neuronChannel, width, height,
						raster, getSourceImageType(), isSourceRasterResize(), isNorm());
				
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "convvae_backpropogate",
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "convvae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	@Override
	public synchronized Raster generateRaster() throws RemoteException {
		try {
			NeuronValue[] dataX = generate();
			if (dataX == null) return null;
			
			if (deconv == null) {
				Dimension size = getConvRasterSize();
				Raster raster = Raster.convertFromNeuronValuesToRaster(dataX, neuronChannel, size.width, size.height,
					getSourceImageType(), isNorm(), getDefaultAlpha());
				
				return raster;
			}
			
			dataX = deconv.evaluate(dataX);
			if (dataX == null) return null;
			
			Dimension size = getDeconvRasterSize();
			return Raster.convertFromNeuronValuesToRaster(dataX, neuronChannel, size.width, size.height,
				getSourceImageType(), isNorm(), getDefaultAlpha());
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;
	}
	
	
	/**
	 * Getting convolutional raster size.
	 * @return convolutional raster size.
	 */
	protected Dimension getConvRasterSize() {
		if (conv == null) return new Dimension(width, height);
		
		ConvLayer convOutputLayer = conv.getConvOutputLayer();
		if (convOutputLayer == null)
			return new Dimension(width, height);
		else
			return new Dimension(convOutputLayer.getWidth(), convOutputLayer.getHeight());
	}
	
	
	/**
	 * Getting deconvolutional raster size.
	 * @return deconvolutional raster size.
	 */
	protected Dimension getDeconvRasterSize() {
		if (deconv == null) return getConvRasterSize();
		
		ConvLayer deconvOutputLayer = deconv.getConvOutputLayer();
		if (deconvOutputLayer == null)
			return getConvRasterSize();
		else
			return new Dimension(deconvOutputLayer.getWidth(), deconvOutputLayer.getHeight());
	}

	
	/**
	 * Getting source image type.
	 * @return source image type.
	 */
	private int getSourceImageType() {
		if (config.containsKey(Raster.SOURCE_IMAGE_TYPE_FIELD))
			return config.getAsInt(Raster.SOURCE_IMAGE_TYPE_FIELD);
		else
			return Raster.SOURCE_IMAGE_TYPE_DEFAULT;
	}
	
	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	private boolean isNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}
	
	
	/**
	 * Getting whether source raster is resized.
	 * @return whether source raster is resized.
	 */
	private boolean isSourceRasterResize() {
		if (config.containsKey(Raster.SOURCE_RESIZE_FIELD))
			return config.getAsBoolean(Raster.SOURCE_RESIZE_FIELD);
		else
			return Raster.SOURCE_RESIZE_DEFAULT;
	}
	
	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	private int getDefaultAlpha() {
		if (config.containsKey(Raster.ALPHA_FIELD))
			return config.getAsInt(Raster.ALPHA_FIELD);
		else
			return Raster.ALPHA_DEFAULT;
	}
	

	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		try (ConvVAEImpl convVAE = new ConvVAEImpl(3, 250, 250)) {
			convVAE.config.put(LEARN_MAX_ITERATION_FIELD, 1);
			
			Filter[] convFilters = new Filter[] {convVAE.getFilterFactory().zoomOut(3, 3)};
			Filter[] deconvFilters = new Filter[] {convVAE.getFilterFactory().zoomIn(3, 3)};
			//convVAE.initialize(10, new int[] {30, 20});
			convVAE.initialize(10, new int[] {30, 20}, convFilters, deconvFilters);
		
			List<Raster> sample = Util.newList(0);
			
			Raster image = Raster.load(Paths.get("working/sample1.png"));
			sample.add(image);
			
			image = Raster.load(Paths.get("working/sample2.png"));
			sample.add(image);

			//image = Raster.load(Paths.get("working/sample3.png"));
			//sample.add(image);

			convVAE.learnByRaster(sample);
			
			image = convVAE.generateRaster();
			image.save(Paths.get("working/gen1.png"));
			
			image = convVAE.generateRaster();
			image.save(Paths.get("working/gen2.png"));
			
			//image = convVAE.generateImage();
			//image.save(Paths.get("working/gen3.png"));
			
			//image = convVAE.generateImage();
			//image.save(Paths.get("working/gen4.png"));
		}
		catch (Exception e) {
			Util.trace(e);
		}
	}


}
