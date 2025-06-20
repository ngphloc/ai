/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.core.generator;

import net.ea.ann.core.generator.Generator;
import net.ea.ann.core.generator.TrainerAbstract;

/**
 * This class is abstract implementation of standard trainer algorithm..
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class TrainerStandardAbstract extends TrainerAbstract implements TrainerStandard {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with generator.
	 * @param generator specified generator.
	 */
	public TrainerStandardAbstract(Generator generator) {
		super(generator);
	}

	
}
