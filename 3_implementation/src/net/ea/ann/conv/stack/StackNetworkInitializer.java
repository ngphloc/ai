/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.io.Serializable;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.raster.Size;

/**
 * This class is initializer of convolutional stack network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class StackNetworkInitializer implements Cloneable, Serializable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal network.
	 */
	protected StackNetworkAbstract network = null;
	
	
	/**
	 * Constructor with specified stack network.
	 * @param network specified stack network.
	 */
	public StackNetworkInitializer(StackNetworkAbstract network) {
		this.network = network;
	}

	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filters specific filters.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		return network.initialize(size, filters, nFullHiddenOutputNeuron, initReverse);
	}
	
	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filters specific filters.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters,
			int[] nFullHiddenOutputNeuron) {
		return initialize(size, filters, nFullHiddenOutputNeuron, false);
	}
	
	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filters specific filters.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters,
			boolean initReverse) {
		return initialize(size, filters, null, initReverse);
	}

	
	/**
	 * Initialize with image/raster specification without reversed fully connected network.
	 * @param size stack content size.
	 * @param filters specific filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters) {
		return initialize(size, filters, null, false);
	}

	
	/**
	 * Initialize with image/raster specification.
	 * @param size stack content size.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		return initialize(size, (Filter[])null, nFullHiddenOutputNeuron, initReverse);
	}
	
	
	/**
	 * Initialize with image/raster specification.
	 * @param size stack content size.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			int[] nFullHiddenOutputNeuron) {
		return initialize(size, (Filter[])null, nFullHiddenOutputNeuron, false);
	}

	
	/**
	 * Initialize with image/raster specification.
	 * @param size stack content size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size) {
		return initialize(size, (Filter[])null, null, false);
	}

	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filterArrays arrays of filters. Filters in the same array have the same size.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[][] filterArrays,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		return network.initialize(size, filterArrays, nFullHiddenOutputNeuron, initReverse);
	}
	
	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filterArrays array of filters. Filters in the same row have the same size.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[][] filterArrays,
			int[] nFullHiddenOutputNeuron) {
		return initialize(size, filterArrays, nFullHiddenOutputNeuron, false);
	}
	
	
	/**
	 * Initialize with image/raster specification.
	 * @param size stack content size.
	 * @param filterArrays array of filters. Filters in the same row have the same size.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[][] filterArrays,
			boolean initReverse) {
		return initialize(size, filterArrays, null, initReverse);
	}

	
	/**
	 * Initialize with image/raster specification without reversed fully connected network.
	 * @param size stack content size.
	 * @param filterArrays array of filters. Filters in the same row have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[][] filterArrays) {
		return initialize(size, filterArrays, null, false);
	}

	
}
