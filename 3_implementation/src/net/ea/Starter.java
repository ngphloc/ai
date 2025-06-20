/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea;

/**
 * Starter program for evolution algorithm package.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Starter {

	
	/**
	 * Default constructor.
	 */
	public Starter() {
		super();
	}

	
	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		try {
			Object ins = Class.forName("net.hudup.Starter").getDeclaredConstructor().newInstance();
			Object param = new String[] {};
			ins.getClass().getDeclaredMethod("main", String[].class).invoke(ins, param);
		} catch (Throwable e) {System.out.println("Cannot load net.hudup.Starter by error " + e.getMessage());}
	}
	

}
