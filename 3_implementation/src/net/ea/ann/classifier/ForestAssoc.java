/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.io.Serializable;

import net.ea.ann.mane.MatrixNetworkAssoc;
import net.ea.ann.transformer.TransformerAssoc;

/**
 * This class provides utility methods for forest classifier.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ForestAssoc implements Cloneable, Serializable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Forest classifier.
	 */
	protected ForestClassifier forest = null;
	
	
	/**
	 * Constructor with forest classifier.
	 * @param forest forest classifier.
	 */
	public ForestAssoc(ForestClassifier forest) {
		this.forest = forest;
	}


	/**
	 * Getting size of parameters.
	 * @return size of parameters.
	 */
	public int sizeOfParams() {
		if (!forest.validate()) return 0;
		int size = 0;
		for (ClassifierAbstract tree : forest.trees) {
			if (tree instanceof MatrixClassifier) {
				MatrixClassifier mac = (MatrixClassifier)tree;
				size += new MatrixNetworkAssoc(mac.nut).sizeOfParams();
				if (mac.adjuster != null) size += new MatrixNetworkAssoc(mac.adjuster).sizeOfParams();
			}
			else if (tree instanceof TransformerClassifier) {
				TransformerClassifier tramac = (TransformerClassifier)tree;
				size += new TransformerAssoc(tramac.transformer).sizeOfParams();
			}
		}
		return size;
	}
	
	
	/**
	 * Getting depth.
	 * @return depth.
	 */
	public int depth() {
		int depth = 0;
		if (!forest.validate()) return 0;
		for (ClassifierAbstract tree : forest.trees) {
			if (tree instanceof MatrixClassifier)
				depth = ((MatrixClassifier)tree).nut.size() - 1;
			else if (tree instanceof TransformerClassifier)
				depth = new TransformerAssoc(((TransformerClassifier)tree).transformer).depth();
		}
		return depth;
	}
	
	
}
