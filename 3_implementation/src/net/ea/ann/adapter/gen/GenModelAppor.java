/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen;

import net.hudup.core.client.PowerServer;
import net.hudup.core.logistic.console.Console;
import net.hudup.core.logistic.console.ConsoleApp;
import net.hudup.core.logistic.console.ConsoleAppor;

/**
 * This class represents generative model application creator.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GenModelAppor extends ConsoleAppor {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Test application creator name.
	 */
	public final static String NAME = "genaitester";
	
	
	/**
	 * Default application creator.
	 */
	public GenModelAppor() {
		super();
	}

	
	@Override
	public String getName() {
		return NAME;
	}


	@Override
	protected ConsoleApp newApp(PowerServer server, ConsoleAppor consoleAppor, Console console) {
		return new GenModelApp(server, (GenModelAppor)consoleAppor, console);
	}


}
