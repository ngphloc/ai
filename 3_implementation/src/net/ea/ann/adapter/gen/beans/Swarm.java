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
 * This class is an extensive implementation of classifier within PSO.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Swarm extends ClassifierModelAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public Swarm() {
		super();
	}
	

	@Override
	public String getName() {
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "classifier.swarm";
	}

	
	@Override
	protected Classifier createGenModel() {
		try {
			return new net.ea.ann.classifier.Swarm(getNeuronChannel(), getRasterChannel());
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
}
