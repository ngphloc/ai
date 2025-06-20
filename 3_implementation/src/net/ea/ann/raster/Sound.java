/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.io.Serializable;
import java.nio.file.Path;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represent a sound record.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Sound extends Cloneable, Serializable {


	/**
	 * Name of default sound format.
	 */
	String SOUND_FORMAT_DEFAULT = "wav";

	
	/**
	 * Getting sound length.
	 * @return sound length.
	 */
	int getLength();


	/**
	 * Save sound record to path.
	 * @param path specified path.
	 * @return true if writing is successful.
	 */
	boolean save(Path path);
	

	/**
	 * Extracting sound record into neuron value array.
	 * @param neuronChannel neuron channel.
	 * @param length sound length.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return neuron value array.
	 */
	NeuronValue[] convertFromSoundToNeuronValues(int neuronChannel, int length, boolean isNorm);


	/**
	 * Getting default format of image.
	 * @return default format of image.
	 */
	static String getDefaultFormat() {
		return SOUND_FORMAT_DEFAULT;
	}


}
