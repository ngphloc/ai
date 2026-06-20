/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen.beans;

import net.ea.ann.adapter.gen.IRAbstract;
import net.ea.ann.core.Util;

/**
 * This class represents the default information retrieval (IR) system.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IRProxyNCA extends IRAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Default constructor.
	 */
	public IRProxyNCA() {
		super();
	}

	
	@Override
	public String getName() {
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "ir.proxynca";
	}

	
	@Override
	protected net.ea.ann.ir.IRDefault createIR() {
		try {
			return new net.ea.ann.ir.IRProxyNCA();
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
}
