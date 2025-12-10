/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.BiasFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.Filter1D;
import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.conv.filter.Filter3D;
import net.ea.ann.conv.filter.Filter4D;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.NeuronValueRaster;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This interface represents a single convolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvLayerSingle extends ConvLayer {

	
	/**
	 * Getting filter.
	 * @return internal filter.
	 */
	Filter getFilter();
	
	
	/**
	 * Setting filter.
	 * @param filter specified filter.
	 * @return previous filter.
	 */
	Filter setFilter(Filter filter);
	
	
	/**
	 * Checking whether padding zero when filtering.
	 * @return whether padding zero when filtering.
	 */
	boolean isPadZeroFilter();
	
	
	/**
	 * Setting whether to pad zero when filtering.
	 * @param isPadZeroFilter flag to indicate whether to pad zero when filtering.
	 */
	void setPadZeroFilter(boolean isPadZeroFilter);
	
	
	/**
	 * Getting bias.
	 * @return bias.
	 */
	NeuronValue getBias();
	
	
	/**
	 * Setting bias
	 * @param bias specified bias.
	 * @return true if setting is successful.
	 */
	boolean setBias(NeuronValue bias);
	
	
	/**
	 * Getting neuron at specific index.
	 * @param index specified index.
	 * @return neuron at specific index.
	 */
	ConvNeuron get(int index);

	
	/**
	 * Setting neuron value at specific index.
	 * @param index specific index.
	 * @param value neuron value.
	 * @return previous neuron value.
	 */
	NeuronValue set(int index, NeuronValue value);
	
	
	/**
	 * Getting layer width.
	 * @return layer width.
	 */
	int getWidth();
	
	
	/**
	 * Getting layer height.
	 * @return layer height.
	 */
	int getHeight();
	
	
	/**
	 * Getting layer depth.
	 * @return layer depth.
	 */
	int getDepth();


	/**
	 * Getting layer time.
	 * @return layer time.
	 */
	int getTime();


	/**
	 * Getting size of neurons.
	 * @return size of neurons.
	 */
	int length();
	
	
	/**
	 * Getting data as array of neurons.
	 * @return data as array of neurons.
	 */
	ConvNeuron[] getNeurons();
	
	
	/**
	 * Getting data as array of neuron value.
	 * @return data as array of neuron value.
	 */
	NeuronValue[] getData();
	
	
	/**
	 * Setting data.
	 * @param data data as array of neuron value.
	 * @return data to be set.
	 */
	NeuronValue[] setData(NeuronValue[] data);
	
	
	/**
	 * Getting reference to activation function.
	 * @return reference to activation function.
	 */
	Function getActivateRef();
	
	
	/**
	 * Setting reference to activation function.
	 * @param activateRef reference to activation function.
	 * @return previous function reference.
	 */
	Function setActivateRef(Function activateRef);


	/**
	 * Creating layer.
	 * @param size size.
	 * @return new layer.
	 */
	ConvLayerSingle newLayer(Size size);
	
	
	/**
	 * Forwarding to evaluate the next layer.
	 * @param nextLayer next layer.
	 * @param filter specified filter.
	 * @return the next layer.
	 */
	ConvLayerSingle forward(ConvLayerSingle nextLayer, Filter filter);

	
	/**
	 * Back-warding errors.
	 * @param nextLayerErrors errors at next layer.
	 * @param learningRate learning rate.
	 * @return errors.
	 */
	ConvLayerSingle[] backward(ConvLayerSingle[] nextLayerErrors, double learningRate);

	
	/**
	 * Back-warding errors.
	 * @param nextLayerErrors errors at next layer.
	 * @param learningRate learning rate.
	 * @param outputBiasFilter output.
	 * @return errors.
	 */
	ConvLayerSingle[] backward(ConvLayerSingle[] nextLayerErrors, double learningRate, BiasFilter outputBiasFilter);

		
	/**
	 * Calculating derivative of this filter given next layer as bias layer at specified coordinator.
	 * @param nextError next layer as next bias.
	 * @param filter specified filter.
	 * @return differentials of kernel.
	 */
	NeuronValue[][] dKernel(ConvLayerSingle nextError, Filter filter);
	
	
	/**
	 * Calculating derivative of this layer given next layer as bias layer at specified coordinator.
	 * @param nextError next layer as next bias.
	 * @param filter specified filter.
	 * @return differentials of values.
	 */
	NeuronValueRaster dValue(ConvLayerSingle nextError, Filter filter);
	
	
	/**
	 * Create raster from neuron values.
	 * @param values neuron values.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return raster created from neuron values.
	 */
	Raster createRaster(NeuronValue[] values,
		boolean isNorm, int defaultAlpha);


	/**
	 * Learning filter from this layer and next layer.
	 * @param initialFilter initial filter. It can be null. This is initial filter of larger layer.
	 * @param learningBias flag to indicate whether to lean bias.
	 * @param learningRate learning rate.
	 * @param maxIteration maximum iteration.
	 * @return filter learned from this layer and next layer.
	 */
	BiasFilter learnFilter(BiasFilter initialFilter, boolean learningBias, double learningRate, int maxIteration);

	
	/**
	 * Back-warding errors.
	 * @param thisLayer this layer.
	 * @param nextLayerErrors errors at next layer.
	 * @param learningRate learning rate.
	 * @param outputBiasFilter output.
	 * @return errors.
	 */
	static ConvLayerSingle[] backward(ConvLayerSingle thisLayer, ConvLayerSingle[] nextLayerErrors, double learningRate, BiasFilter outputBiasFilter) {
		ConvLayerSingle[] outputErrors = new ConvLayerSingle2D[nextLayerErrors.length]; 
		NeuronValue[] dFilterErrors = new NeuronValue[nextLayerErrors.length];
		NeuronValue[][][] dFilterKernels = new NeuronValue[nextLayerErrors.length][][];
		NeuronValue zero = thisLayer.newNeuronValue().zero();
		for (int i = 0; i < nextLayerErrors.length; i++) {
			NeuronValueRaster dValues = thisLayer.dValue(nextLayerErrors[i], outputBiasFilter.filter);
			outputErrors[i] = thisLayer.newLayer(new Size(thisLayer.getWidth(), thisLayer.getHeight(), 1, 1));
			outputErrors[i].setData(dValues.getValues());
			NeuronValue valueSum = NeuronValue.valueSum(outputErrors[i].getData());
			dFilterErrors[i] = Filter.CALC_ERROR_MEAN ?
				(dValues.getCountValues() > 0 ? valueSum.divide(dValues.getCountValues()) : zero) :
				(valueSum); //Filter errors.
			dFilterKernels[i] = thisLayer.dKernel(nextLayerErrors[i], outputBiasFilter.filter); //Filter kernel errors.
		}
		
		NeuronValue dfilterBiasMean = NeuronValue.valueMean(dFilterErrors);
		NeuronValue filterBias = thisLayer.getBias().add(dfilterBiasMean.multiply(learningRate));
		outputBiasFilter.bias = filterBias; //Update filter bias.
		
		if (outputBiasFilter.filter instanceof Filter4D) {
			
		}
		else if (outputBiasFilter.filter instanceof Filter3D) {
			
		}
		else if (outputBiasFilter.filter instanceof Filter2D) {
			NeuronValue[][] dfilterKernelMean = Filter2D.kernelMean(dFilterKernels);
			dfilterKernelMean = NeuronValue.multiply(dfilterKernelMean, learningRate);
			Filter2D filter = ((Filter2D)outputBiasFilter.filter).shallowClone();
			filter.accumKernel(dfilterKernelMean);
			outputBiasFilter.filter = filter; //Update filter.
		}
		else if (outputBiasFilter.filter instanceof Filter1D) {
			
		}

		return outputErrors;
	}
	
	
}
