/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen.beans;

import net.ea.ann.adapter.gen.ClassifierModelAbstract;
import net.ea.ann.classifier.Classifier;
import net.ea.ann.core.Util;

/**
 * This class is default implementation of classifier within context of matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixClassifier extends ClassifierModelAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public MatrixClassifier() {
		super();
	}
	

	@Override
	public String getName() {
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "classifier.mac";
	}

	
	@Override
	protected Classifier createGenModel() {
		try {
			return net.ea.ann.classifier.MatrixClassifier.create(getRasterChannel(), isNorm());
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
}
