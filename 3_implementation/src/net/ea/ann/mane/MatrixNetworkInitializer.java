/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.awt.Dimension;
import java.io.Serializable;
import java.util.Arrays;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.NetworkStandard;

/**
 * This utility class provides initialization methods for matrix neural network.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixNetworkInitializer implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal matrix neural network.
	 */
	protected MatrixNetworkImpl mane = null;
	
	
	/**
	 * Constructor with matrix neural network.
	 * @param mane matrix neural network.
	 */
	public MatrixNetworkInitializer(MatrixNetworkImpl mane) {
		this.mane = mane;
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension outputSize2, int depth2) {
		return mane.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2);
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filterStride1 filter stride 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Dimension filterStride1, int depth1, boolean dual1, Dimension outputSize2, int depth2) {
		return mane.initialize(inputSize1, outputSize1, filterStride1, depth1, dual1, outputSize2, depth2);
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension outputSize2, int depth2) {
		return initialize(inputSize1, null, filter1, depth1, dual1, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Filter2D filter1, int depth1, Dimension outputSize2, int depth2) {
		return initialize(inputSize1, filter1, depth1, false, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, int depth1, Dimension outputSize2, int depth2) {
		return initialize(inputSize1, (Filter2D)null, depth1, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension outputSize2) {
		return initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, 0);
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, boolean dual1, Dimension outputSize2, int depth2) {
		return initialize(inputSize1, outputSize1, filter1, 0, dual1, outputSize2, depth2);
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, boolean dual1, Dimension outputSize2) {
		return initialize(inputSize1, outputSize1, filter1, dual1, outputSize2, 0);
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, Dimension outputSize2) {
		return initialize(inputSize1, outputSize1, filter1, false, outputSize2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, int depth1, Dimension outputSize2, int depth2) {
		return initialize(inputSize1, outputSize1, (Filter2D)null, depth1, false, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Dimension outputSize2) {
		return initialize(inputSize1, outputSize1, null, outputSize2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, boolean dual1, int depth2) {
		return initialize(inputSize1, outputSize1, filter1, dual1, null, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size.
	 * @param outputSize1 output size, which can be null.
	 * @param filter1 filter, which can be null.
	 * @param dual1 dual mode.
	 * @param outputSize2 output size, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Dimension outputSize, Filter2D filter, boolean dual) {
		return initialize(inputSize, outputSize, filter, dual, null);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @param dual dual mode.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Dimension outputSize, Filter2D filter, int depth, boolean dual) {
		return initialize(inputSize, outputSize, filter, depth, dual, null);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @param dual dual mode.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Dimension outputSize, Filter2D filter, int depth) {
		return initialize(inputSize, outputSize, filter, depth, false);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Dimension outputSize, Filter2D filter) {
		return initialize(inputSize, outputSize, filter, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Dimension outputSize, int depth) {
		return initialize(inputSize, outputSize, null, depth);
	}

		
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Dimension outputSize) {
		return initialize(inputSize, outputSize, (Filter2D)null);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @param dual dual mode.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Filter2D filter, int depth, boolean dual) {
		return initialize(inputSize, null, filter, depth);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Filter2D filter, int depth) {
		return initialize(inputSize, filter, depth, false);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, Filter2D filter) {
		return initialize(inputSize, filter, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize, int depth) {
		return initialize(inputSize, (Dimension)null, depth);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize) {
		return initialize(inputSize, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, Dimension outputSize2, int depth2) {
		return initialize(inputSize1, outputSize1, filter1, depth1, true, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be null.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Dimension inputSize1, Filter2D filter1, int depth1, Dimension outputSize2, int depth2) {
		return initializeDual(inputSize1, null, filter1, depth1, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, Dimension outputSize2) {
		return initializeDual(inputSize1, outputSize1, filter1, depth1, outputSize2, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, Dimension outputSize2) {
		return initializeDual(inputSize1, outputSize1, filter1, 0, outputSize2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Dimension inputSize, Dimension outputSize, Filter2D filter, int depth) {
		return initializeDual(inputSize, outputSize, filter, depth, null);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Dimension inputSize, Dimension outputSize, Filter2D filter) {
		return initializeDual(inputSize, outputSize, filter, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Dimension inputSize, Filter2D filter, int depth) {
		return initializeDual(inputSize, null, filter, depth);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Dimension inputSize, Filter2D filter) {
		return initializeDual(inputSize, filter, 0);
	}

	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param hBase height base, which can be 0.
	 * @param wBase width base, which can be 0.
	 * @param depth depth, which can be 0.
	 * @return hidden neuron numbers.
	 */
	static int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize, Dimension outputSize, int hBase, int wBase, int depth) {
		if (inputSize == null) return null;
		if (inputSize.width <= 0 || inputSize.height <= 0) return null;
		hBase = hBase < 2 ? MatrixNetworkImpl.BASE_DEFAULT : hBase;
		wBase = wBase < 2 ? MatrixNetworkImpl.BASE_DEFAULT : wBase;
		
		if (outputSize == null || outputSize.width <= 0 || outputSize.height <= 0) {
			int H = inputSize.height, W = inputSize.width;
			if (depth <= 0) {
				depth = Math.min(H, W) / Math.max(hBase, wBase);
				depth = Math.min(depth, MatrixNetworkImpl.DEPTH_DEFAULT);
			}
			else
				depth = Math.min(depth,  Math.min(H, W) / Math.max(hBase, wBase));
			if (depth <= 0) return null;
			
			int[] heights = null, widths = null;
			for (int i = 0; i < depth; i++) {
				H = (H-hBase)/hBase + 1;
				W = (W-wBase)/wBase + 1;
				if (H < hBase || W < wBase) break;
				
				if (heights == null)
					heights = new int[] {H};
				else {
					heights = Arrays.copyOf(heights, heights.length+1);
					heights[heights.length-1] = H;
				}
				if (widths == null)
					widths = new int[] {W};
				else {
					widths = Arrays.copyOf(widths, widths.length+1);
					widths[widths.length-1] = W;
				}
			}
			
			return new int[][] {heights, widths};
		}
		
		//Calculating hidden layer number.
		int[] hHiddens = NetworkStandard.constructHiddenNeuronNumbers(inputSize.height, outputSize.height, hBase, 0);
		int[] wHiddens = NetworkStandard.constructHiddenNeuronNumbers(inputSize.width, outputSize.width, wBase, 0);
		int[] heights = null, widths = null;
		if ( (hHiddens == null || hHiddens.length == 0) && (wHiddens == null || wHiddens.length == 0) ) {
			heights = new int[] {outputSize.height};
			widths = new int[] {outputSize.width};
		}
		else {
			if (hHiddens == null || hHiddens.length == 0) {
				heights = new int[wHiddens.length];
				Arrays.fill(heights, outputSize.height);
				widths = wHiddens;
			}
			else if (wHiddens == null || wHiddens.length == 0) {
				heights = hHiddens;
				widths = new int[hHiddens.length];
				Arrays.fill(widths, outputSize.width);
			}
			else if (hHiddens.length < wHiddens.length) {
				heights = Arrays.copyOf(hHiddens, wHiddens.length);
				Arrays.fill(heights, hHiddens.length, heights.length, heights[hHiddens.length-1]);
				widths = wHiddens;
			}
			else if (hHiddens.length > wHiddens.length) {
				heights = hHiddens;
				widths = Arrays.copyOf(wHiddens, hHiddens.length);
				Arrays.fill(widths, wHiddens.length, widths.length, widths[wHiddens.length-1]);
			}
			else {
				heights = hHiddens;
				widths = wHiddens;
			}
			
			heights = Arrays.copyOf(heights, heights.length+1);
			heights[heights.length-1] = outputSize.height;
			widths = Arrays.copyOf(widths, widths.length+1);
			widths[widths.length-1] = outputSize.width;
		}
		if (heights.length != widths.length) return null;
		
		//Filling depth.
		if (depth > heights.length) {
			int length = heights.length;
			int d = depth / length;
			int r = depth % length;
			int[] newHeights = new int[length*d + r];
			int[] newWidths = new int[length*d + r];
			
			for (int i = 0; i < length; i++) {
				int index = i*d;
				Arrays.fill(newHeights, index, index+d, heights[i]);
				Arrays.fill(newWidths, index, index+d, widths[i]);
			}
			if (r > 0) {
				int index = length*d;
				Arrays.fill(newHeights, index, index+r, newHeights[index-1]);
				Arrays.fill(newWidths, index, index+r, newWidths[index-1]);
			}
			heights = newHeights;
			widths = newWidths;
		}
		return new int[][] {heights, widths};
	}

	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param hBase height base, which can be 0.
	 * @param wBase width base, which can be 0.
	 * @return hidden neuron numbers.
	 */
	static int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize, Dimension outputSize, int hBase, int wBase) {
		return constructHiddenOutputNeuronNumbers(inputSize, outputSize, hBase, wBase, 0);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param hBase height base, which can be 0.
	 * @param wBase width base, which can be 0.
	 * @param depth depth, which can be 0.
	 * @return hidden neuron numbers.
	 */
	static int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize, int hBase, int wBase, int depth) {
		return constructHiddenOutputNeuronNumbers(inputSize, null, hBase, wBase, depth);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param hBase height base, which can be 0.
	 * @param wBase width base, which can be 0.
	 * @return hidden neuron numbers.
	 */
	static int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize, int hBase, int wBase) {
		return constructHiddenOutputNeuronNumbers(inputSize, hBase, wBase, 0);
	}

	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @return hidden neuron numbers.
	 */
	static int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize) {
		return constructHiddenOutputNeuronNumbers(inputSize, 0, 0);
	}

	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @param depth depth, which can be 0.
	 * @return hidden neuron numbers.
	 */
	int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize, Dimension outputSize, Filter2D filter, int depth) {
		int hBase = filter != null ? filter.getStrideHeight() : MatrixNetworkImpl.BASE_DEFAULT;
		int wBase = filter != null ? filter.getStrideWidth() : MatrixNetworkImpl.BASE_DEFAULT;
		return constructHiddenOutputNeuronNumbers(inputSize, outputSize, hBase, wBase, depth);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @return hidden neuron numbers.
	 */
	int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize, Dimension outputSize, Filter2D filter) {
		return constructHiddenOutputNeuronNumbers(inputSize, outputSize, filter, 0);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @param depth depth, which can be 0.
	 * @return hidden neuron numbers.
	 */
	int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize, Filter2D filter, int depth) {
		return constructHiddenOutputNeuronNumbers(inputSize, null, filter, depth);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @return hidden neuron numbers.
	 */
	int[][] constructHiddenOutputNeuronNumbers(Dimension inputSize, Filter2D filter) {
		return constructHiddenOutputNeuronNumbers(inputSize, filter, 0);
	}

	
}
