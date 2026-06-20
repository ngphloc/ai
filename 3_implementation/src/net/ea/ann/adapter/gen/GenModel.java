/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen;

import net.ea.ann.adapter.Delegator;
import net.hudup.core.alg.ExecuteAsLearnAlg;

/**
 * This interface is the most abstract interface for generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface GenModel extends GenModelRemoteTask, ExecuteAsLearnAlg, Delegator {

	
}
