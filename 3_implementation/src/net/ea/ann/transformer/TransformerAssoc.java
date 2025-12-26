/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.io.Serializable;

import net.ea.ann.mane.MatrixNetworkAssoc;

/**
 * This utility class provides utility methods for default transformer.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerAssoc implements Cloneable, Serializable {


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
	public TransformerAssoc(TransformerImpl transformer) {
		this.transformer = transformer;
	}

	
	/**
	 * Getting size of attention parameters.
	 * @param attention attention.
	 * @return size of attention parameters.
	 */
	private static int sizeOfParams(Attention attention) {
		if (attention == null || !attention.validate()) return 0;
		int size = 0;
		if (attention.WO != null) size += attention.WO.rows()*attention.WO.columns();
		for (int i = 0; i < attention.h(); i++) {
			Attention0 head = attention.head(i);
			if (head.WQ != null) size += head.WQ.rows()*head.WQ.columns();
			if (head.WK != null) size += head.WK.rows()*head.WK.columns();
			if (head.WV != null) size += head.WV.rows()*head.WV.columns();
			if (head.M != null) size += head.M.length*head.M[0].length;
			if (head.T1 != null) size += head.T1.rows()*head.T1.columns();
			if (head.T2 != null) size += head.T2.rows()*head.T2.columns();
		}
		return size;
	}

	
	/**
	 * Getting size of block parameters.
	 * @param block transformer block.
	 * @return size of block parameters.
	 */
	private static int sizeOfParams(TransformerBlock block) {
		if (!block.validate()) return 0;
		int size = sizeOfParams(block.attention);
		if (block.ffn != null) size += new MatrixNetworkAssoc(block.ffn).sizeOfParams();
		if (block.outputAdapter != null) size += new MatrixNetworkAssoc(block.outputAdapter).sizeOfParams();
		return size;
	}
	
	
	/**
	 * Getting size of basic transformer parameters.
	 * @return size of basic transformer parameters.
	 */
	static int sizeOfParams(TransformerBasic transformer) {
		if (!transformer.validate()) return 0;
		int size = 0;
		for (int i = 0; i < transformer.size(); i++) size += sizeOfParams(transformer.get(i));
		return size;
	}
	
	
	/**
	 * Getting size of transformer parameters.
	 * @return size of transformer parameters.
	 */
	public int sizeOfParams() {
		if (!transformer.validate()) return 0;
		int size = 0;
		if (transformer.encoder != null && transformer.decoder != null)
			size += sizeOfParams(transformer.encoder) + sizeOfParams(transformer.decoder);
		else if (transformer.encoder != null && transformer.decoder == null)
			size += sizeOfParams(transformer.encoder);
		else if (transformer.encoder == null && transformer.decoder != null)
			size += sizeOfParams(transformer.decoder);
		return size;
	}
	
	
	





	/**
	 * Getting depth of attention parameters.
	 * @param attention attention.
	 * @return depth of attention parameters.
	 */
	private static int depth(Attention attention) {
		return attention.validate() ? 1 : 0;
	}

	
	/**
	 * Getting depth of block parameters.
	 * @param block transformer block.
	 * @return depth block parameters.
	 */
	private static int depth(TransformerBlock block) {
		if (!block.validate()) return 0;
		int depth = depth(block.attention) + block.ffn.size()-1;
		if (block.outputAdapter != null) depth += block.outputAdapter.size()-1;
		return depth;
	}
	
	
	/**
	 * Getting depth of basic transformer parameters.
	 * @return depth of basic transformer parameters.
	 */
	static int depth(TransformerBasic transformer) {
		if (!transformer.validate()) return 0;
		int depth = 0;
		for (int i = 0; i < transformer.size(); i++) depth += depth(transformer.get(i));
		return depth;
	}
	
	
	/**
	 * Getting depth of transformer parameters.
	 * @return depth of transformer parameters.
	 */
	public int depth() {
		if (!transformer.validate()) return 0;
		int size = 0;
		if (transformer.encoder != null && transformer.decoder != null)
			size += depth(transformer.encoder) + depth(transformer.decoder);
		else if (transformer.encoder != null && transformer.decoder == null)
			size += depth(transformer.encoder);
		else if (transformer.encoder == null && transformer.decoder != null)
			size += depth(transformer.decoder);
		return size;
	}
	
	
}
