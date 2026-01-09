/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.mane.weight.TransformerWeight;
import net.ea.ann.mane.weight.WeightImpl;
import net.ea.ann.raster.Size;

/**
 * This class represents filter specification.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightSpec implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This enum specifies weight type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum Type {
		
		/**
		 * Normal weight.
		 */
		normal,
		
		/**
		 * Transformed-based weight.
		 */
		transformer,
		
		/**
		 * Null weight.
		 */
		nil,
		
	}

	
	/**
	 * Weight type.
	 */
	public Type type = Type.normal;

	
	/**
	 * Default constructor.
	 */
	public WeightSpec(Type type) {
		this.type = type;
	}


	/**
	 * Converting type to integer number.
	 * @param type type.
	 * @return integer number.
	 */
	public static int typeToInt(Type type) {
		return type.ordinal();
	}
	

	/**
	 * Converting integer number to type.
	 * @param typeOrdinal integer number.
	 * @return type.
	 */
	public static Type intToType(int typeOrdinal) {
		Type type = Type.normal;
		switch (typeOrdinal) {
		case 0:
			type = Type.normal;
			break;
		case 1:
			type = Type.transformer;
			break;
		case 2:
			type = Type.nil;
			break;
		default:
			type = Type.normal;
			break;
		}
		return type;
	}
	
	
	/**
	 * Converting string to type.
	 * @param typeText string.
	 * @return type.
	 */
	public static Type stringToType(String typeText) {
		return Type.valueOf(typeText);
	}
	
	
	/**
	 * Creating weight.
	 * @param sizeW1 the first weight size.
	 * @param sizeW2 the second weight size.
	 * @param hint hinting value.
	 * @param layerSpec layer specification, which can be null.
	 * @param neuronChannel neuron channel.
	 */
	public static Weight newWeight(Size sizeW1, Size sizeW2, NeuronValue hint, LayerSpec layerSpec, int neuronChannel) {
		if (sizeW2 != null || layerSpec == null)
			return WeightImpl.create(sizeW1, sizeW2, hint);
		Size prevSize = layerSpec.prevSize, thisSize = layerSpec.size;
		if (prevSize == null || thisSize == null)
			return WeightImpl.create(sizeW1, sizeW2, hint);
		if (prevSize.width != thisSize.width || prevSize.height != thisSize.height)
			return WeightImpl.create(sizeW1, sizeW2, hint);
		if (layerSpec.weightSpec == null || layerSpec.weightSpec.type != Type.transformer)
			return WeightImpl.create(sizeW1, sizeW2, hint);
		
		return TransformerWeight.create(neuronChannel, prevSize, thisSize);
	}


	/**
	 * Creating weight.
	 * @param sizeW1 the first weight size.
	 * @param sizeW2 the second weight size.
	 * @param hint hinting value.
	 */
	public static Weight newWeight(Size sizeW1, Size sizeW2, NeuronValue hint) {
		return WeightImpl.create(sizeW1, sizeW2, hint);
	}


}
