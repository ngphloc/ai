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

import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkStandard;
import net.ea.ann.mane.filter.FilterSpec;
import net.ea.ann.raster.Size;

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
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, int depth1, boolean dual1, Size outputSize2, int depth2) {
		return mane.initialize(inputSize1, outputSize1, filterSpec1, depth1, dual1, outputSize2, depth2);
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize1, FilterSpec filterSpec1, int depth1, boolean dual1, Size outputSize2, int depth2) {
		return initialize(inputSize1, null, filterSpec1, depth1, dual1, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize1, FilterSpec filterSpec1, int depth1, Size outputSize2, int depth2) {
		return initialize(inputSize1, filterSpec1, depth1, false, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize1, int depth1, Size outputSize2, int depth2) {
		return initialize(inputSize1, (FilterSpec)null, depth1, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, int depth1, boolean dual1, Size outputSize2) {
		return initialize(inputSize1, outputSize1, filterSpec1, depth1, dual1, outputSize2, 0);
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
	public boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, boolean dual1, Size outputSize2, int depth2) {
		return initialize(inputSize1, outputSize1, filterSpec1, 0, dual1, outputSize2, depth2);
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, boolean dual1, Size outputSize2) {
		return initialize(inputSize1, outputSize1, filterSpec1, dual1, outputSize2, 0);
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, Size outputSize2) {
		return initialize(inputSize1, outputSize1, filterSpec1, false, outputSize2);
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
	public boolean initialize(Size inputSize1, Size outputSize1, int depth1, Size outputSize2, int depth2) {
		return initialize(inputSize1, outputSize1, (FilterSpec)null, depth1, false, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize1, Size outputSize1, Size outputSize2) {
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
	public boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, boolean dual1, int depth2) {
		return initialize(inputSize1, outputSize1, filterSpec1, dual1, null, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size.
	 * @param outputSize1 output size, which can be null.
	 * @param filterSpec1 filter, which can be null.
	 * @param dual1 dual mode.
	 * @param outputSize2 output size, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize, FilterSpec filter, boolean dual) {
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
	public boolean initialize(Size inputSize, Size outputSize, FilterSpec filter, int depth, boolean dual) {
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
	public boolean initialize(Size inputSize, Size outputSize, FilterSpec filter, int depth) {
		return initialize(inputSize, outputSize, filter, depth, false);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize, FilterSpec filter) {
		return initialize(inputSize, outputSize, filter, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize, int depth) {
		return initialize(inputSize, outputSize, null, depth);
	}

		
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize) {
		return initialize(inputSize, outputSize, (FilterSpec)null);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @param dual dual mode.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, FilterSpec filter, int depth, boolean dual) {
		return initialize(inputSize, null, filter, depth);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, FilterSpec filter, int depth) {
		return initialize(inputSize, filter, depth, false);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, FilterSpec filter) {
		return initialize(inputSize, filter, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, int depth) {
		return initialize(inputSize, (Size)null, depth);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize) {
		return initialize(inputSize, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, int depth1, Size outputSize2, int depth2) {
		return initialize(inputSize1, outputSize1, filterSpec1, depth1, true, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be null.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Size inputSize1, FilterSpec filterSpec1, int depth1, Size outputSize2, int depth2) {
		return initializeDual(inputSize1, null, filterSpec1, depth1, outputSize2, depth2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, int depth1, Size outputSize2) {
		return initializeDual(inputSize1, outputSize1, filterSpec1, depth1, outputSize2, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param outputSize2 output size 1, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, Size outputSize2) {
		return initializeDual(inputSize1, outputSize1, filterSpec1, 0, outputSize2);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Size inputSize, Size outputSize, FilterSpec filter, int depth) {
		return initializeDual(inputSize, outputSize, filter, depth, null);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Size inputSize, Size outputSize, FilterSpec filter) {
		return initializeDual(inputSize, outputSize, filter, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Size inputSize, FilterSpec filter, int depth) {
		return initializeDual(inputSize, null, filter, depth);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeDual(Size inputSize, FilterSpec filter) {
		return initializeDual(inputSize, filter, 0);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filterSpec1 filter specification 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeByDepth(Size inputSize1, Size outputSize1, FilterSpec filterSpec1, int depth1, boolean dual1, Size outputSize2, int depth2) {
		return mane.initializeByDepth(inputSize1, outputSize1, filterSpec1, depth1, dual1, outputSize2, depth2);		
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
	public boolean initializeByDepth(Size inputSize, Size outputSize, FilterSpec filter, int depth, boolean dual) {
		return initializeByDepth(inputSize, outputSize, filter, depth, dual, null, 0);		
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeByDepth(Size inputSize, Size outputSize, FilterSpec filter, int depth) {
		return initializeByDepth(inputSize, outputSize, filter, depth, false);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeByDepth(Size inputSize, Size outputSize, int depth) {
		return initializeByDepth(inputSize, outputSize, (FilterSpec)null, depth);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @param depth the number of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	public boolean initializeByDepth(Size inputSize, int depth) {
		return initializeByDepth(inputSize, null, depth);
	}

	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize input size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeByDepth(Size inputSize) {
		return initializeByDepth(inputSize, MatrixNetworkImpl.DEPTH_DEFAULT);
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
	public static int[][] constructHiddenOutputNeuronNumbers(Size inputSize, Size outputSize, int hBase, int wBase, int depth) {
		if (inputSize == null) return null;
		if (inputSize.width <= 0 || inputSize.height <= 0) return null;
		hBase = hBase < NetworkAbstract.ZOOMOUT_DEFAULT ? NetworkAbstract.ZOOMOUT_DEFAULT : hBase;
		wBase = wBase < NetworkAbstract.ZOOMOUT_DEFAULT ? NetworkAbstract.ZOOMOUT_DEFAULT : wBase;
		
		if (outputSize == null || outputSize.width <= 0 || outputSize.height <= 0) {
			int H = inputSize.height, W = inputSize.width;
			int maxBase = Math.max(Math.max(hBase, wBase), MatrixNetworkImpl.MINSIZE);
			maxBase = Math.min(maxBase, Math.min(H, W));
			if (depth <= 0) {
				depth = Math.min(H, W) / maxBase;
				depth = Math.min(depth, MatrixNetworkImpl.DEPTH_DEFAULT);
			}
			else
				depth = Math.min(depth, Math.min(H, W) / maxBase);
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
	static int[][] constructHiddenOutputNeuronNumbers(Size inputSize, Size outputSize, int hBase, int wBase) {
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
	static int[][] constructHiddenOutputNeuronNumbers(Size inputSize, int hBase, int wBase, int depth) {
		return constructHiddenOutputNeuronNumbers(inputSize, null, hBase, wBase, depth);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param hBase height base, which can be 0.
	 * @param wBase width base, which can be 0.
	 * @return hidden neuron numbers.
	 */
	static int[][] constructHiddenOutputNeuronNumbers(Size inputSize, int hBase, int wBase) {
		return constructHiddenOutputNeuronNumbers(inputSize, hBase, wBase, 0);
	}

	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @return hidden neuron numbers.
	 */
	static int[][] constructHiddenOutputNeuronNumbers(Size inputSize) {
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
	int[][] constructHiddenOutputNeuronNumbers(Size inputSize, Size outputSize, Dimension filter, int depth) {
		int hBase = filter != null ? filter.height : NetworkAbstract.ZOOMOUT_DEFAULT;
		int wBase = filter != null ? filter.width : NetworkAbstract.ZOOMOUT_DEFAULT;
		return constructHiddenOutputNeuronNumbers(inputSize, outputSize, hBase, wBase, depth);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param outputSize output size, which can be null.
	 * @param filter filter, which can be null.
	 * @return hidden neuron numbers.
	 */
	int[][] constructHiddenOutputNeuronNumbers(Size inputSize, Size outputSize, Dimension filter) {
		return constructHiddenOutputNeuronNumbers(inputSize, outputSize, filter, 0);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @param depth depth, which can be 0.
	 * @return hidden neuron numbers.
	 */
	int[][] constructHiddenOutputNeuronNumbers(Size inputSize, Dimension filter, int depth) {
		return constructHiddenOutputNeuronNumbers(inputSize, null, filter, depth);
	}
	
	
	/**
	 * Constructing hidden and output neuron numbers.
	 * @param inputSize input size.
	 * @param filter filter, which can be null.
	 * @return hidden neuron numbers.
	 */
	int[][] constructHiddenOutputNeuronNumbers(Size inputSize, Dimension filter) {
		return constructHiddenOutputNeuronNumbers(inputSize, filter, 0);
	}

	
}
