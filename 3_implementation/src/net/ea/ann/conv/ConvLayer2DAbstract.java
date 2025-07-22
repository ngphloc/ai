/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.awt.Rectangle;
import java.util.Arrays;

import net.ea.ann.conv.filter.BiasFilter;
import net.ea.ann.conv.filter.DeconvConvFilter;
import net.ea.ann.conv.filter.DeconvConvFilter2D;
import net.ea.ann.conv.filter.DeconvFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.conv.filter.ProductFilter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.NeuronRaster;
import net.ea.ann.raster.NeuronValueRaster;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;

/**
 * This class is an abstract implementation of convolutional layer in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ConvLayer2DAbstract extends ConvLayer1DAbstract implements ConvLayerSingle2D {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Raster height.
	 */
	protected int height = 1;
	
	
	/**
	 * Constructor with neuron channel, activation function, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer2DAbstract(int neuronChannel, Function activateRef, int width, int height, Filter filter, Id idRef) {
		this(neuronChannel, activateRef, filter, idRef);
		
		this.width = width;
		this.height = height;
		this.neurons = new ConvNeuron[width*height];
		NeuronValue zero = this.newNeuronValue().zero();
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int index = y*width + x;
				ConvNeuron neuron = this.newNeuron();
				neuron.setValue(zero);
				this.neurons[index] = neuron;
			}
		}
	}


	/**
	 * Constructor with neuron channel, activation function, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 */
	protected ConvLayer2DAbstract(int neuronChannel, Function activateRef, int width, int height, Filter filter) {
		this(neuronChannel, activateRef, width, height, filter, null);
	}


	/**
	 * Constructor with neuron channel, activation function, width, and height.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 */
	protected ConvLayer2DAbstract(int neuronChannel, Function activateRef, int width, int height) {
		this(neuronChannel, activateRef, width, height, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ConvLayer2DAbstract(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}

	
	@Override
	public int getHeight() {
		return height;
	}


	@Override
	public Filter2D getFilter2D() {
		if (filter == null)
			return null;
		else if (filter instanceof Filter2D)
			return (Filter2D)filter;
		else
			return null;
	}

	
	@Override
	public ConvNeuron get(int x, int y) {
		return neurons[y*width + x];
	}


	@Override
	public NeuronValue set(int x, int y, NeuronValue value) {
		ConvNeuron neuron = neurons[y*width + x];
		if (neuron == null)
			return null;
		else {
			NeuronValue prevValue = neuron.getValue();
			neuron.setValue(value);
			return prevValue;
		}
	}


	@Override
	protected NeuronValue[] getData(Rectangle region) {
		if (region == null) return getData();
		int width = getWidth();
		int height = getHeight();
		
		region.x = region.x < 0 ? 0 : region.x;
		region.y = region.y < 0 ? 0 : region.y;
		region.width = region.x + region.width <= width ? region.width : width - region.x;
		region.height = region.y + region.height <= height ? region.height : height - region.y;
		if (region.width <= 0 || region.height <= 0) return null;
		
		int regionIndex = 0;
		NeuronValue[] data = new NeuronValue[region.width*region.height];
		int yheight = region.y + region.height;
		int xwidth = region.x + region.width;
		for (int y = region.y; y < yheight; y++) {
			for (int x = region.x; x < xwidth; x++) {
				int index = y*width + x;
				data[regionIndex] = neurons[index].getValue();
				regionIndex++;
			}
		}
		
		return data;
	}
	

	@Override
	protected NeuronValue[] setData(NeuronValue[] data, Rectangle region) {
		if (region == null) setData(data);
		if (data == null || neurons.length == 0) return null;
		int width = getWidth();
		int height = getHeight();
		
		region.x = region.x < 0 ? 0 : region.x;
		region.y = region.y < 0 ? 0 : region.y;
		region.width = region.x + region.width <= width ? region.width : width - region.x;
		region.height = region.y + region.height <= height ? region.height : height - region.y;
		if (region.width <= 0 || region.height <= 0) return null;
		
		int regionIndex = 0;
		data = NeuronValue.adjustArray(data, region.width*region.height, this);
		int yheight = region.y + region.height;
		int xwidth = region.x + region.width;
		for (int y = region.y; y < yheight; y++) {
			int yw = y*width;
			for (int x = region.x; x < xwidth; x++) {
				int index = yw + x;
				neurons[index].setValue(data[regionIndex]);
				regionIndex++;
			}
		}

		return data;
	}

	
	/**
	 * Getting the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 * @param x X coordination.
	 * @param y Y coordination.
	 * @return the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 */
	protected Rectangle getNextRegion(int x, int y) {
		Rectangle nextRegion = super.getNextRegion(x);
		if (nextRegion == null) return null;
		Filter filter = getFilter();

		int filterStrideHeight = filter.getStrideHeight();
		int nextHeight = ((ConvLayerSingle)nextLayer).getHeight();
		
		if (filter instanceof DeconvFilter) {
			nextRegion.y = y * filterStrideHeight;
			nextRegion.height = filterStrideHeight;
		}
		else {
			nextRegion.y = y / filterStrideHeight;
			nextRegion.height = 1;
		}
		
		nextRegion.height = nextRegion.y + nextRegion.height <= nextHeight ? nextRegion.height : nextHeight - nextRegion.y;
		if (nextRegion.height <= 0)
			return null;
		else
			return nextRegion;
	}
	
	
	@Override
	protected Rectangle getNextRegion(Rectangle thisRegion) {
		thisRegion = thisRegion != null ? thisRegion : new Rectangle(getWidth(), getHeight());
		Rectangle nextRegion = super.getNextRegion(thisRegion);
		if (nextRegion == null) return null;
		Filter filter = getFilter();
		
		int filterStrideHeight = filter.getStrideHeight();
		int nextHeight = ((ConvLayerSingle)nextLayer).getHeight();
		
		if (filter instanceof DeconvFilter) {
			nextRegion.y = thisRegion.y * filterStrideHeight;
			nextRegion.height = thisRegion.height * filterStrideHeight;
		}
		else {
			nextRegion.y = thisRegion.y / filterStrideHeight;
			nextRegion.height = thisRegion.height / filterStrideHeight;
			nextRegion.height = nextRegion.height < 1 ? 1 : nextRegion.height;
		}
		
		nextRegion.height = nextRegion.y + nextRegion.height <= nextHeight ? nextRegion.height : nextHeight - nextRegion.y;
		if (nextRegion.height <= 0)
			return null;
		else
			return nextRegion;
	}
	
	
	/**
	 * Getting the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 * @param x X coordination.
	 * @param y Y coordination.
	 * @return the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 */
	protected Rectangle getPrevRegion(int x, int y) {
		Rectangle prevRegion = super.getPrevRegion(x);
		if (prevRegion == null) return null;
		ConvLayerSingle prevLayer = (ConvLayerSingle)this.prevLayer;
		Filter filter = prevLayer.getFilter();
		
		int filterStrideHeight = filter.getStrideHeight();
		int prevHeight = prevLayer.getHeight();
		int prevBlockHeight = filter.isMoveStride() ? prevHeight / filterStrideHeight : prevHeight;
		
		if (filter instanceof DeconvFilter) {
			prevRegion.y = y / filterStrideHeight;
			prevRegion.y = prevRegion.y < prevHeight ? prevRegion.y : prevHeight-1;

			prevRegion.height = 1;
		}
		else {
			int yBlock = y < prevBlockHeight ? y : prevBlockHeight-1;
			prevRegion.y = yBlock*filterStrideHeight;
			
			prevRegion.height = filterStrideHeight;
		}
		
		prevRegion.height = prevRegion.y + prevRegion.height <= prevHeight ? prevRegion.height : prevHeight - prevRegion.y;
		if (prevRegion.height <= 0)
			return null;
		else
			return prevRegion;
	}
	
	
	@Override
	protected Rectangle getPrevRegion(Rectangle thisRegion) {
		thisRegion = thisRegion != null ? thisRegion : new Rectangle(getWidth(), getHeight());
		Rectangle prevRegion = super.getPrevRegion(thisRegion);
		if (prevRegion == null) return null;
		ConvLayerSingle prevLayer = (ConvLayerSingle)this.prevLayer;
		Filter filter = prevLayer.getFilter();
		
		int filterStrideHeight = filter.getStrideHeight();
		int prevHeight = prevLayer.getHeight();
		int prevBlockHeight = filter.isMoveStride() ? prevHeight / filterStrideHeight : prevHeight;
		
		if (filter instanceof DeconvFilter) {
			prevRegion.y = thisRegion.y / filterStrideHeight;
			prevRegion.y = prevRegion.y < prevHeight ? prevRegion.y : prevHeight-1;

			prevRegion.height = thisRegion.height / filterStrideHeight;
			prevRegion.height = prevRegion.height < 1 ? 1 : prevRegion.height;
		}
		else {
			int yBlock = thisRegion.y < prevBlockHeight ? thisRegion.y : prevBlockHeight-1;
			prevRegion.y = yBlock*filterStrideHeight;
			
			prevRegion.height = thisRegion.height*filterStrideHeight;
		}
		
		prevRegion.height = prevRegion.y + prevRegion.height <= prevHeight ? prevRegion.height : prevHeight - prevRegion.y;
		if (prevRegion.height <= 0)
			return null;
		else
			return prevRegion;
	}

	
	@Override
	public ConvLayer forward() {
		ConvLayer nextLayer = getNextLayer();
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle2D)) return null;
		NeuronRaster result = forward(this, (ConvLayerSingle2D)nextLayer, getFilter(), null, null, true);
		return result != null ? nextLayer : null;
	}

	
	@Override
	public ConvLayerSingle forward(ConvLayerSingle nextLayer, Filter filter) {
		NeuronRaster result = forward(this, (ConvLayerSingle2D)nextLayer, filter, null, null, true);
		return result != null ? nextLayer : null;
	}


	@Override
	public NeuronValue[][] dKernel(ConvLayerSingle nextError, Filter filter) {
		return dKernel(this, (ConvLayerSingle2D)nextError, filter, null, null);
	}


	@Override
	public NeuronValueRaster dValue(ConvLayerSingle nextError, Filter filter) {
		return dValue(this, (ConvLayerSingle2D)nextError, filter, null, null);
	}


	/**
	 * Forwarding evaluation from current layer to next layer.
	 * @param thisLayer current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @param nextAffected flag to indicate whether the next layer is affected.
	 * @return arrays of neurons filtered.
	 */
	static NeuronRaster forward(ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer, Filter f, Rectangle thisFilterRegion, Rectangle nextFilterRegion, boolean nextAffected) {
		if (thisLayer == null) return null;
		Filter2D filter = (f != null && f instanceof Filter2D) ? (Filter2D)f : thisLayer.getFilter2D();
		if (filter == null) {
			return ConvLayer1DAbstract.forward(thisLayer, nextLayer, f, thisFilterRegion, nextFilterRegion, nextAffected);
		}

		NeuronValue nextZero = nextLayer != null ? nextLayer.newNeuronValue().zero() : thisLayer.newNeuronValue().zero();
		ConvNeuron[] nextNeurons = nextLayer.getNeurons();
		if (nextAffected)
			nextNeurons = nextLayer.getNeurons();
		else {
			nextNeurons = new ConvNeuron[nextLayer.length()];
			for (int i = 0; i < nextNeurons.length; i++) {
				nextNeurons[i].setValue(nextZero);
				nextNeurons[i].setInput(nextZero);
			}
		}
		if (nextNeurons == null || nextNeurons.length == 0) return null;
		
		if (filter instanceof DeconvConvFilter) {
			for (ConvNeuron nextNeuron : nextNeurons) {
				nextNeuron.setValue(null);
				nextNeuron.setInput(null);
			}
		}
		
		if (thisFilterRegion != null && nextFilterRegion != null) nextFilterRegion = null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int filterStrideHeight = filter.getStrideHeight();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int thisBlockHeight = filter.isMoveStride() ? thisHeight / filterStrideHeight : thisHeight;
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		Function activateRef = nextLayer.getActivateRef();
		activateRef = activateRef == null ? thisLayer.getActivateRef() : activateRef;
		
		for (int nextY = 0; nextY < nextHeight; nextY++) {
			int thisY = 0;
			if (filter instanceof DeconvFilter) {
				thisY = nextY / filterStrideHeight;
				if (!nextLayer.isPadZeroFilter()) thisY = thisY < thisHeight ? thisY : thisHeight-1;
			}
			else {
				int yBlock = nextLayer.isPadZeroFilter() ? nextY : (nextY < thisBlockHeight ? nextY : thisBlockHeight-1);
				thisY = yBlock*filterStrideHeight;
			}
			
			int nextIndexY = nextY*nextWidth;
			for (int nextX = 0; nextX < nextWidth; nextX++) {
				int thisX = 0;
				if (filter instanceof DeconvFilter) {
					thisX = nextX / filterStrideWidth;
					if (!nextLayer.isPadZeroFilter()) thisX = thisX < thisWidth ? thisX : thisWidth-1;
				}
				else {
					int xBlock = nextLayer.isPadZeroFilter() ? nextX : (nextX < thisBlockWidth ? nextX : thisBlockWidth-1);
					thisX = xBlock*filterStrideWidth;
				}
				
				//Setting zero to outline pixels.
				int nextIndex = nextIndexY + nextX;
				if (thisY >= thisHeight || thisX >= thisWidth) {
					nextNeurons[nextIndex].setValue(nextZero);
					continue;
				}
				
				//Checking region.
				if (thisFilterRegion != null && !thisFilterRegion.contains(thisX, thisY)) continue;
				if (nextFilterRegion != null && !nextFilterRegion.contains(nextX, nextY)) continue;
				
				//Filtering
				NeuronValue filteredValue = null;
				if (filter instanceof DeconvConvFilter)
					filteredValue = ((DeconvConvFilter2D)filter).apply(thisX, thisY, thisLayer, nextX, nextY, nextLayer);
				else
					filteredValue = filter.apply(thisX, thisY, thisLayer);
				
				if (filteredValue != null) {
					filteredValue = filteredValue.add(thisLayer.getBias());
					nextNeurons[nextIndex].setInput(filteredValue);
					if (activateRef != null) filteredValue = activateRef.evaluate(filteredValue);
					nextNeurons[nextIndex].setValue(filteredValue);
				}
				else
					nextNeurons[nextIndex].setValue(nextZero);
			}
		}
		
		if (filter instanceof DeconvConvFilter) {
			for (ConvNeuron nextNeuron : nextNeurons) {
				if (nextNeuron.getValue() == null) nextNeuron.setValue(nextZero);
				if (nextNeuron.getInput() == null) nextNeuron.setInput(nextZero);
			}
		}

		if ((!(thisLayer instanceof ConvLayer2DAbstract)) || (thisFilterRegion == null && nextFilterRegion == null))
			return new NeuronRaster(nextLayer.getNeuronChannel(), nextNeurons, new Size(nextWidth, nextHeight, 1, 1));
		
		Rectangle nextRegion = null;
		if (thisFilterRegion != null)
			nextRegion = ((ConvLayer2DAbstract)thisLayer).getNextRegion(thisFilterRegion);
		else
			nextRegion = nextFilterRegion;
		if (nextRegion == null) return new NeuronRaster(nextLayer.getNeuronChannel(), nextNeurons, new Size(nextWidth, nextHeight, 1, 1));
		
		ConvNeuron[] regionNeurons = new ConvNeuron[nextRegion.width*nextRegion.height];
		int regionIndex = 0;
		for (int nextY = nextRegion.y; nextY < nextRegion.y + nextRegion.height; nextY++) {
			int nextLength = nextY*nextWidth;
			for (int nextX = nextRegion.x; nextX < nextRegion.x + nextRegion.width; nextX++) {
				int nextIndex = nextLength + nextX;
				regionNeurons[regionIndex] = nextNeurons[nextIndex];
				regionIndex++;
			}
		}
		return new NeuronRaster(nextLayer.getNeuronChannel(), regionNeurons, new Size(nextRegion.width, nextRegion.height, 1, 1));
	}


	/**
	 * Forwarding evaluation from current layer to next layer.
	 * @param input input data. It can be null.
	 * @param thisLayer current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @param nextAffected flag to indicate whether the next layer is affected
	 * @return arrays of neurons filtered.
	 */
	public static NeuronRaster forward(NeuronValue[] input, ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer, Filter f, Rectangle thisFilterRegion, Rectangle nextFilterRegion, boolean nextAffected) {
		if ((input != null) && (thisLayer instanceof ConvLayer2DAbstract)) ((ConvLayer2DAbstract)thisLayer).setData(input);
		return forward(thisLayer, nextLayer, f, thisFilterRegion, nextFilterRegion, nextAffected);
	}
	
	
	/**
	 * Forwarding evaluation from current layer to next layer.
	 * @param thisLayer current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @return arrays of neurons filtered.
	 */
	public static NeuronRaster forward(ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer, Filter f) {
		return forward(thisLayer, nextLayer, f, null, null, true);
	}
	
	
	/**
	 * Calculating derivative of this filter given next layer as bias layer at specified coordinator.
	 * @param thisLayerSize current layer size.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @return differentials of kernel.
	 */
	static NeuronValue[][] dKernel(ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer, Filter f, Rectangle thisFilterRegion, Rectangle nextFilterRegion) {
		if (thisLayer == null || nextLayer == null) return null;
		Filter2D filter = (f != null && f instanceof Filter2D) ? (Filter2D)f : thisLayer.getFilter2D();
		if (filter == null)
			return ConvLayer1DAbstract.dKernel(thisLayer, nextLayer, f, thisFilterRegion, nextFilterRegion);
		if (filter instanceof DeconvConvFilter)
			throw new RuntimeException("Derivative not implemented with de-convolutional filter yet");
		if (!(filter instanceof ProductFilter2D))
			throw new RuntimeException("Derivative not implemented with non-product filter yet");
		
		NeuronValue thisZero = thisLayer != null ? thisLayer.newNeuronValue().zero() : nextLayer.newNeuronValue().zero();;
		NeuronValue[][] thisKernel = new NeuronValue[filter.height()][filter.width()];
		for (int i = 0; i < thisKernel.length; i++) {
			for (int j = 0; j < thisKernel[i].length; j++) thisKernel[i][j] = thisZero;
		}
		
		if (thisFilterRegion != null && nextFilterRegion != null) nextFilterRegion = null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int filterStrideHeight = filter.getStrideHeight();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int thisBlockHeight = filter.isMoveStride() ? thisHeight / filterStrideHeight : thisHeight;
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		
		int countKernel = 0;
		for (int nextY = 0; nextY < nextHeight; nextY++) {
			int thisY = 0;
			if (filter instanceof DeconvFilter) {
				thisY = nextY / filterStrideHeight;
				if (!nextLayer.isPadZeroFilter()) thisY = thisY < thisHeight ? thisY : thisHeight-1;
			}
			else {
				int yBlock = nextLayer.isPadZeroFilter() ? nextY : (nextY < thisBlockHeight ? nextY : thisBlockHeight-1);
				thisY = yBlock*filterStrideHeight;
			}
			
			for (int nextX = 0; nextX < nextWidth; nextX++) {
				int thisX = 0;
				if (filter instanceof DeconvFilter) {
					thisX = nextX / filterStrideWidth;
					if (!nextLayer.isPadZeroFilter()) thisX = thisX < thisWidth ? thisX : thisWidth-1;
				}
				else {
					int xBlock = nextLayer.isPadZeroFilter() ? nextX : (nextX < thisBlockWidth ? nextX : thisBlockWidth-1);
					thisX = xBlock*filterStrideWidth;
				}
				
				//Ignoring outline pixels.
				if (thisY >= thisHeight || thisX >= thisWidth) continue;

				//Checking region.
				if (thisFilterRegion != null && !thisFilterRegion.contains(thisX, thisY)) continue;
				if (nextFilterRegion != null && !nextFilterRegion.contains(nextX, nextY)) continue;
				
				//Calculating derivative.
				NeuronValue[][] dKernel = null;
				dKernel = filter.dKernel(nextX, nextY, thisLayer, nextLayer);
				if (dKernel == null) continue;
				
				for (int i = 0; i < dKernel.length; i++) {
					for (int j = 0; j < dKernel[i].length; j++) {
						thisKernel[i][j] = thisKernel[i][j].add(dKernel[i][j]);
					}
				}
				countKernel++;
			}
		}
		
		//Calculating mean of kernel.
		if (countKernel > 0) {
			for (int i = 0; i < thisKernel.length; i++) {
				for (int j = 0; j < thisKernel[i].length; j++)
					thisKernel[i][j] = thisKernel[i][j].divide(countKernel);
			}
		}
		return thisKernel;
	}

	
	/**
	 * Calculating derivative of this layer given next layer as bias layer at specified coordinator.
	 * @param thisLayerSize current layer size.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @return differentials of values.
	 */
	static NeuronValueRaster dValue(ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer, Filter f, Rectangle thisFilterRegion, Rectangle nextFilterRegion) {
		if (thisLayer == null || nextLayer == null) return null;
		Filter2D filter = (f != null && f instanceof Filter2D) ? (Filter2D)f : thisLayer.getFilter2D();
		if (filter == null)
			return ConvLayer1DAbstract.dValue(thisLayer, nextLayer, f, thisFilterRegion, nextFilterRegion);
		if (filter instanceof DeconvConvFilter)
			throw new RuntimeException("Derivative not implemented with de-convolutional filter yet");

		NeuronValue thisZero = thisLayer != null ? thisLayer.newNeuronValue().zero() : nextLayer.newNeuronValue().zero();;
		NeuronValue[] thisValues = new NeuronValue[thisLayer.getWidth()*thisLayer.getHeight()];
		for (int i = 0; i < thisValues.length; i++) thisValues[i] = thisZero;
		int[] thisValuesCount = new int[thisValues.length];
		Arrays.fill(thisValuesCount, 0);
		
		if (thisFilterRegion != null && nextFilterRegion != null) nextFilterRegion = null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int filterStrideHeight = filter.getStrideHeight();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int thisBlockHeight = filter.isMoveStride() ? thisHeight / filterStrideHeight : thisHeight;
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		
		int countValues = 0;
		for (int nextY = 0; nextY < nextHeight; nextY++) {
			int thisY = 0;
			if (filter instanceof DeconvFilter) {
				thisY = nextY / filterStrideHeight;
				if (!nextLayer.isPadZeroFilter()) thisY = thisY < thisHeight ? thisY : thisHeight-1;
			}
			else {
				int yBlock = nextLayer.isPadZeroFilter() ? nextY : (nextY < thisBlockHeight ? nextY : thisBlockHeight-1);
				thisY = yBlock*filterStrideHeight;
			}
			
			for (int nextX = 0; nextX < nextWidth; nextX++) {
				int thisX = 0;
				if (filter instanceof DeconvFilter) {
					thisX = nextX / filterStrideWidth;
					if (!nextLayer.isPadZeroFilter()) thisX = thisX < thisWidth ? thisX : thisWidth-1;
				}
				else {
					int xBlock = nextLayer.isPadZeroFilter() ? nextX : (nextX < thisBlockWidth ? nextX : thisBlockWidth-1);
					thisX = xBlock*filterStrideWidth;
				}
				
				//Ignoring outline pixels.
				if (thisY >= thisHeight || thisX >= thisWidth) continue;

				//Checking region.
				if (thisFilterRegion != null && !thisFilterRegion.contains(thisX, thisY)) continue;
				if (nextFilterRegion != null && !nextFilterRegion.contains(nextX, nextY)) continue;
				
				//Calculating derivative.
				NeuronValue[][] dValues = null;
				dValues = filter.dValue(nextX, nextY, thisLayer, nextLayer);
				if (dValues == null) continue;
				
				for (int i = 0; i < dValues.length; i++) {
					int rowLength = (thisY+i) * thisWidth;
					for (int j = 0; j < dValues[i].length; j++) {
						int index = rowLength + (thisX+j);
						if (index >= thisValues.length) continue; //Not so necessary.
						thisValues[index] = thisValues[index].add(dValues[i][j]);
						thisValuesCount[index] = thisValuesCount[index] + 1;
					}
				}
			}
		}
		
		//Calculating mean of values.
		for (int i = 0; i < thisValues.length; i++) {
			if (thisValuesCount[i] <= 0) continue;
			thisValues[i] = thisValues[i].divide((double)thisValuesCount[i]);
			countValues++;
		}
		
		if (thisFilterRegion == null && nextFilterRegion == null)
			return new NeuronValueRaster(nextLayer.getNeuronChannel(), thisValues, new Size(thisWidth, thisHeight, 1, 1), countValues);

		Rectangle thisRegion = null;
		if (nextFilterRegion != null)
			thisRegion = ((ConvLayer2DAbstract)nextLayer).getPrevRegion(nextFilterRegion);
		else
			thisRegion = thisFilterRegion;
		if (thisRegion == null)
			return new NeuronValueRaster(nextLayer.getNeuronChannel(), thisValues, new Size(thisWidth, thisHeight, 1, 1), countValues);

		NeuronValue[] regionValues = new NeuronValue[thisRegion.width*thisRegion.height];
		int regionIndex = 0;
		for (int thisY = thisRegion.y; thisY < thisRegion.y + thisRegion.height; thisY++) {
			int thisLength = thisY*thisWidth;
			for (int thisX = thisRegion.x; thisX < thisRegion.x + thisRegion.width; thisX++) {
				int thisIndex = thisLength + thisX;
				regionValues[regionIndex] = thisValues[thisIndex];
				regionIndex++;
			}
		}
		return new NeuronValueRaster(nextLayer.getNeuronChannel(), regionValues, new Size(thisRegion.width, thisRegion.height, 1, 1), countValues);
	}

	
	@Override
	public Raster createRaster(NeuronValue[] values,
			boolean isNorm, int defaultAlpha) {
		return RasterAssoc.createRaster(this, values, isNorm, defaultAlpha);
	}


	@Override
	public BiasFilter learnFilter(BiasFilter initialFilter, boolean learningBias, double learningRate, int maxIteration) {
		if (nextLayer == null) return null;
		ConvLayerSingle2D smallLayer = null, largeLayer = null;
		if (this.length() >= ((ConvLayerSingle)nextLayer).length()) {
			smallLayer = (ConvLayerSingle2D)nextLayer;
			largeLayer = this;
		}
		else {
			smallLayer = this;
			largeLayer = (ConvLayerSingle2D)nextLayer;
		}
		
		maxIteration = maxIteration > 0 ? maxIteration : Network.LEARN_MAX_ITERATION_DEFAULT;
		int iteration = 0;
		BiasFilter filter = initialFilter;
		while (iteration < maxIteration) {
			double lr = NetworkAbstract.calcLearningRate(learningRate, iteration, NetworkAbstract.LEARN_RATE_FIXED_DEFAULT);
			filter = learnFilter(smallLayer, largeLayer, filter, learningBias, lr);
			iteration++;
		}
		
		return filter;
	}


	/**
	 * Learning product filter from large layer to small layer.
	 * @param smallLayer specified small layer.
	 * @param largeLayer specified large layer.
	 * @param initialFilter initial filter. This is initial filter of larger layer.
	 * @param learningBias flag to indicate whether to learn bias.
	 * @param learningRate learning rate.
	 * @return filter learned from large layer to small layer. It is filter stored in large layer.
	 */
	static BiasFilter learnFilter(ConvLayerSingle2D smallLayer, ConvLayerSingle2D largeLayer, BiasFilter initialFilter, boolean learningBias, double learningRate) {
		if (smallLayer == null || largeLayer == null) return null;
		if (largeLayer.getHeight() <= 1) return ConvLayer1DAbstract.learnFilter(smallLayer, largeLayer, initialFilter, learningBias, learningRate);
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? 1 : learningRate;

		ConvLayerSingle2D thisLayer = largeLayer, nextLayer = smallLayer;
		int n = Math.min(largeLayer.getWidth()/smallLayer.getWidth(), largeLayer.getHeight()/smallLayer.getHeight());
		boolean isMoveStride = true;
		if (n <= 1) {
			n = 3;
			isMoveStride = false;
		}
		
		NeuronValue zero = thisLayer.newNeuronValue().zero();
		ProductFilter2D filter = null;
		ProductFilter2D initialProductFilter = ((initialFilter != null) && (initialFilter.filter != null) && (initialFilter.filter instanceof ProductFilter2D)) ?
			(ProductFilter2D)(initialFilter.filter) : null;
		if (initialProductFilter == null || initialProductFilter.width() != n || initialProductFilter.height() != n) {
			NeuronValue[][] kernel = new NeuronValue[n][n];
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) kernel[i][j] = zero;
			}
			filter = ProductFilter2D.create(kernel, zero.unit());
		}
		else {
			NeuronValue[][] kernel = new NeuronValue[n][n];
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) kernel[i][j] = initialProductFilter.getKernel()[i][j];
			}
			filter = ProductFilter2D.create(kernel, initialProductFilter.getWeight());
		}
		
		filter.setMoveStride(isMoveStride);
		NeuronValue[][] kernel = filter.getKernel();
		
		NeuronValue bias = null;
		if ((initialFilter != null) && (initialFilter.bias != null))
			bias = initialFilter.bias;
		else
			bias = zero; 
		
		int filterStrideWidth = filter.getStrideWidth();
		int filterStrideHeight = filter.getStrideHeight();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int thisBlockHeight = filter.isMoveStride() ? thisHeight / filterStrideHeight : thisHeight;
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
	
		for (int nextY = 0; nextY < nextHeight-1; nextY++) { //Please pay attention to minus one (-1) which prevents overlapping estimation.
			int thisY = 0;
			int yBlock = nextY < thisBlockHeight ? nextY : thisBlockHeight-1;
			thisY = yBlock*filterStrideHeight;
			
			for (int nextX = 0; nextX < nextWidth-1; nextX++) { //Please pay attention to minus one (-1) which prevents overlapping estimation.
				int thisX = 0;
				int xBlock = nextX < thisBlockWidth ? nextX : thisBlockWidth-1;
				thisX = xBlock*filterStrideWidth;
				
				NeuronValue filteredNextValue = filter.apply(thisX, thisY, thisLayer);
				if (filteredNextValue == null) continue;
				Function nextActivateRef = nextLayer.getActivateRef(); 
				if (nextActivateRef != null) filteredNextValue = nextActivateRef.evaluate(filteredNextValue); 
				
				NeuronValue realNextValue = nextLayer.get(nextX, nextY).getValue();
				NeuronValue error = realNextValue.subtract(filteredNextValue);
				if (nextActivateRef != null)
					error = error.multiplyDerivative(nextActivateRef.derivative(filteredNextValue));
				
				//Learning kernel.
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
						NeuronValue thisValue = thisLayer.get(thisX+j, thisY+i).getValue();
						NeuronValue delta = error.multiply(thisValue).multiply(learningRate);
						kernel[i][j] = kernel[i][j].add(delta);
					}
				}
				
				//Learning bias.
				if (learningBias) bias = bias.add(error.multiply(learningRate));
			}
		}
	
		return new BiasFilter(filter, learningBias?bias:null);
	}
	
	
}
