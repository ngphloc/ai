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
	 * @param outputHead output head.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks, MatrixNetworkImpl outputHead) {
		if (!transformer.initialize(h, n, dm, dk, dv, ffnDepth, nBlocks)) return false;
		return outputHead != null ? transformer.setOutputHead(outputHead) : true;
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
	 * @param headOutputSize head output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks, Dimension headOutputSize) {
		if (!transformer.initialize(h, n, dm, dk, dv, ffnDepth, nBlocks)) return false;
		return headOutputSize != null ? transformer.setOutputHead(headOutputSize, ffnDepth) : true;
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
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputHead output head.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, int nBlocks, MatrixNetworkImpl outputHead) {
		return initialize(1, n, dm, dm, dm, ffnDepth, nBlocks, outputHead);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param headOutputSize head output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, int nBlocks, Dimension headOutputSize) {
		return initialize(1, n, dm, dm, dm, ffnDepth, nBlocks, headOutputSize);
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
	 * @param outputHead output head.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, MatrixNetworkImpl outputHead) {
		return initialize(n, dm, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputHead);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param headOutputSize head output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int ffnDepth, Dimension headOutputSize) {
		return initialize(n, dm, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, headOutputSize);
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
	 * @param outputHead output head.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, MatrixNetworkImpl outputHead) {
		return initialize(n, dm, MatrixNetworkImpl.DEPTH_DEFAULT, outputHead);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param headOutputSize head output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, Dimension headOutputSize) {
		return initialize(n, dm, MatrixNetworkImpl.DEPTH_DEFAULT, headOutputSize);
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
	 * @param outputHead output head.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks, MatrixNetworkImpl outputHead) {
		if (!transformer.initializeOnlyEncoder(h, n, dm, dk, dv, ffnDepth, nBlocks)) return false;
		return outputHead != null ? transformer.setOutputHead(outputHead) : true;
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
	 * @param headOutputSize head output size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks, Dimension headOutputSize) {
		if (!transformer.initializeOnlyEncoder(h, n, dm, dk, dv, ffnDepth, nBlocks)) return false;
		return headOutputSize != null ? transformer.setOutputHead(headOutputSize, ffnDepth) : true;
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
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param outputHead output head.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, int nBlocks, MatrixNetworkImpl outputHead) {
		return initializeOnlyEncoder(1, n, dm, dm, dm, ffnDepth, nBlocks, outputHead);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link TransformerBasic#BLOCKS_NUMBER_DEFAULT}
	 * @param headOutputSize head output size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, int nBlocks, Dimension headOutputSize) {
		return initializeOnlyEncoder(1, n, dm, dm, dm, ffnDepth, nBlocks, headOutputSize);
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
	 * @param outputHead output head.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, MatrixNetworkImpl outputHead) {
		return initializeOnlyEncoder(n, dm, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, outputHead);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param headOutputSize head output size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, int ffnDepth, Dimension headOutputSize) {
		return initializeOnlyEncoder(n, dm, ffnDepth, TransformerBasic.BLOCKS_NUMBER_DEFAULT, headOutputSize);
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
	 * @param outputHead output head.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, MatrixNetworkImpl outputHead) {
		return initializeOnlyEncoder(n, dm, MatrixNetworkImpl.DEPTH_DEFAULT, outputHead);
	}


	/**
	 * Initializing transformer.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param headOutputSize head output size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int n, int dm, Dimension headOutputSize) {
		return initializeOnlyEncoder(n, dm, MatrixNetworkImpl.DEPTH_DEFAULT, headOutputSize);
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
