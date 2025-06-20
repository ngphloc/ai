/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.io.Serializable;

import net.ea.ann.core.NetworkConfig;

/**
 * This class represent setting or parameters of convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvGenSetting implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Name of width field.
	 */
	public final static String WIDTH_FIELD = "convgen_width";
	
	
	/**
	 * Name of height field.
	 */
	public final static String HEIGHT_FIELD = "convgen_height";

	
	/**
	 * Name of depth field.
	 */
	public final static String DEPTH_FIELD = "convgen_depth";

	
	/**
	 * Name of time field.
	 */
	public final static String TIME_FIELD = "convgen_time";

	
	/**
	 * Name of thick-stack field.
	 */
	public final static String THICK_STACK_FIELD = "convgen_thick_stack";

	
	/**
	 * Default value of thick-stack field.
	 */
	public final static boolean THICK_STACK_DEFAULT = false;

	
	/**
	 * Width.
	 */
	public int width = 1;
	
	
	/**
	 * Height.
	 */
	public int height = 1;
	
	
	/**
	 * Depth.
	 */
	public int depth = 1;

	
	/**
	 * Time.
	 */
	public int time = 1;

	
	/**
	 * Thick-stack property. In thick-stack mode, every stack should have more than one element layer.
	 */
	public boolean thickStack = THICK_STACK_DEFAULT;
	
	
	/**
	 * Default construction.
	 */
	public ConvGenSetting() {

	}

	
	/**
	 * Extracting from configuration.
	 * @param config network configuration.
	 */
	public void extractConfig(NetworkConfig config) {
		if (config == null) return;
		
		if (config.containsKey(WIDTH_FIELD)) width = config.getAsInt(WIDTH_FIELD);
		if (config.containsKey(HEIGHT_FIELD)) height = config.getAsInt(HEIGHT_FIELD);
		if (config.containsKey(DEPTH_FIELD)) depth = config.getAsInt(DEPTH_FIELD);
		if (config.containsKey(TIME_FIELD)) time = config.getAsInt(TIME_FIELD);
		if (config.containsKey(THICK_STACK_FIELD)) thickStack = config.getAsBoolean(THICK_STACK_FIELD);
	}
	
	
	
}
