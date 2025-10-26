/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.awt.Dimension;
import java.io.Serializable;

import net.ea.ann.mane.MatrixNetworkImpl;

/**
 * This utility class provides initialization methods for default transformer.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerInitializer implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal transformer.
	 */
	protected TransformerImpl transformer = null;

	
	/**
	 * Constructor with transformer.
	 * @param transformer transformer.
	 */
	public TransformerInitializer(TransformerImpl transformer) {
		this.transformer = transformer;
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks, MatrixNetworkImpl outputAdapter) {
		if (!transformer.initialize(h, n, dm, dk, dv, ffnDepth, nBlocks)) return false;
		return outputAdapter != null ? transformer.setOutputAdapter(outputAdapter) : true;
	}
	
	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks, Dimension outputAdapterOutputSize) {
		if (!transformer.initialize(h, n, dm, dk, dv, ffnDepth, nBlocks)) return false;
		return outputAdapterOutputSize != null ? transformer.setOutputAdapter(outputAdapterOutputSize, ffnDepth) : true;
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
		return initialize(h, n, dm, dk, dv, ffnDepth, nBlocks, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, MatrixNetworkImpl outputAdapter) {
		return initialize(h, n, dm, dk, dv, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputAdapter);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, Dimension outputAdapterOutputSize) {
		return initialize(h, n, dm, dk, dv, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth) {
		return initialize(h, n, dm, dk, dv, ffnDepth, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int ffnDepth, int nBlocks, MatrixNetworkImpl outputAdapter) {
		if (h < 1) return false;
		int dk = dm/h;
		if (dk < 1) return false;
		dm = dk*h;
		return initialize(h, n, dm, dk, dk, ffnDepth, nBlocks, outputAdapter);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int ffnDepth, int nBlocks, Dimension outputAdapterOutputSize) {
		if (h < 1) return false;
		int dk = dm/h;
		if (dk < 1) return false;
		dm = dk*h;
		return initialize(h, n, dm, dk, dk, ffnDepth, nBlocks, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int ffnDepth, int nBlocks) {
		return initialize(h, n, dm, ffnDepth, nBlocks, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, int nBlocks, MatrixNetworkImpl outputAdapter) {
		return initialize(1, n, dm, dm, dm, ffnDepth, nBlocks, outputAdapter);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, int nBlocks, Dimension outputAdapterOutputSize) {
		return initialize(1, n, dm, dm, dm, ffnDepth, nBlocks, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, int nBlocks) {
		return initialize(n, dm, ffnDepth, nBlocks, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, MatrixNetworkImpl outputAdapter) {
		return initialize(n, dm, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputAdapter);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, Dimension outputAdapterOutputSize) {
		return initialize(n, dm, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth) {
		return initialize(n, dm, ffnDepth, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, MatrixNetworkImpl outputAdapter) {
		return initialize(n, dm, MatrixNetworkImpl.DEPTH_DEFAULT, outputAdapter);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, Dimension outputAdapterOutputSize) {
		return initialize(n, dm, MatrixNetworkImpl.DEPTH_DEFAULT, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm) {
		return initialize(n, dm, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks, MatrixNetworkImpl outputAdapter) {
		if (!transformer.initializeOnlyEncoder(h, n, dm, dk, dv, ffnDepth, nBlocks)) return false;
		return outputAdapter != null ? transformer.setOutputAdapter(outputAdapter) : true;
	}


	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks, Dimension outputAdapterOutputSize) {
		if (!transformer.initializeOnlyEncoder(h, n, dm, dk, dv, ffnDepth, nBlocks)) return false;
		return outputAdapterOutputSize != null ? transformer.setOutputAdapter(outputAdapterOutputSize, ffnDepth) : true;
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
		return initializeOnlyEncoder(h, n, dm, dk, dv, ffnDepth, nBlocks, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth, MatrixNetworkImpl outputAdapter) {
		return initialize(h, n, dm, dk, dv, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputAdapter);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth, Dimension outputAdapterOutputSize) {
		return initialize(h, n, dm, dk, dv, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth) {
		return initialize(h, n, dm, dk, dv, ffnDepth, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int ffnDepth, int nBlocks, MatrixNetworkImpl outputAdapter) {
		if (h < 1) return false;
		int dk = dm/h;
		if (dk < 1) return false;
		dm = dk*h;
		return initialize(h, n, dm, dk, dk, ffnDepth, nBlocks, outputAdapter);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int ffnDepth, int nBlocks, Dimension outputAdapterOutputSize) {
		if (h < 1) return false;
		int dk = dm/h;
		if (dk < 1) return false;
		dm = dk*h;
		return initialize(h, n, dm, dk, dk, ffnDepth, nBlocks, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int ffnDepth, int nBlocks) {
		return initialize(h, n, dm, ffnDepth, nBlocks, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, int nBlocks, MatrixNetworkImpl outputAdapter) {
		return initializeOnlyEncoder(1, n, dm, dm, dm, ffnDepth, nBlocks, outputAdapter);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, int nBlocks, Dimension outputAdapterOutputSize) {
		return initializeOnlyEncoder(1, n, dm, dm, dm, ffnDepth, nBlocks, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, int nBlocks) {
		return initializeOnlyEncoder(n, dm, ffnDepth, nBlocks, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, MatrixNetworkImpl outputAdapter) {
		return initializeOnlyEncoder(n, dm, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputAdapter);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, Dimension outputAdapterOutputSize) {
		return initializeOnlyEncoder(n, dm, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth) {
		return initializeOnlyEncoder(n, dm, ffnDepth, (MatrixNetworkImpl)null);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param outputAdapter output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, MatrixNetworkImpl outputAdapter) {
		return initializeOnlyEncoder(n, dm, MatrixNetworkImpl.DEPTH_DEFAULT, outputAdapter);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, Dimension outputAdapterOutputSize) {
		return initializeOnlyEncoder(n, dm, MatrixNetworkImpl.DEPTH_DEFAULT, outputAdapterOutputSize);
	}

	
	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm) {
		return initializeOnlyEncoder(n, dm, (MatrixNetworkImpl)null);
	}


}
