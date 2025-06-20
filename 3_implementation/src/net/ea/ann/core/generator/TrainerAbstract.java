/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.generator;

/**
 * This class is abstract implementation of trainer algorithm..
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class TrainerAbstract implements Trainer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal generator.
	 */
	protected Generator generator = null;
	
	
	/**
	 * Constructor with generator.
	 * @param generator specified generator.
	 */
	public TrainerAbstract(Generator generator) {
		setGenerator(generator);
	}

	
	@Override
	public void setGenerator(Generator generator) {
		if (generator != null) this.generator = generator;
	}


}
