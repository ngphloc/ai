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
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.mane.weight.ActivateFWeight;
import net.ea.ann.mane.weight.ActivateWWeight;
import net.ea.ann.mane.weight.NullWeight;
import net.ea.ann.mane.weight.TransformerWeight;
import net.ea.ann.mane.weight.WeightImpl;
import net.ea.ann.mane.weight.WeightNetworkImpl;
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
		 * Kernel weight.
		 */
		kernel,
		
		/**
		 * Network-based weight.
		 */
		network,
	}
	
	
	/**
	 * This enum represents kernel type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum KernelType {
		
		/**
		 * Normal kernel.
		 */
		normal,
		
		/**
		 * Transformer-based kernel.
		 */
		transformer,
		
		/**
		 * Filter activation type which is like null type but having filter activation function.
		 */
		filter_activate,
		
		/**
		 * Weight activation type which is like null type but having weight activation function.
		 */
		weight_activate,
		
		/**
		 * Null type.
		 */
		nil,
	}

	
	/**
	 * This enum represents network-based weight type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum NetworkType {
		
		/**
		 * Basic type.
		 */
		basic,
		
	}

	
	/**
	 * Weight type.
	 */
	public Type type = Type.kernel;

	
	/**
	 * Kernel type.
	 */
	public KernelType kernelType = KernelType.normal;
	
	
	/**
	 * Network type.
	 */
	public NetworkType networkType = NetworkType.basic;
	
	
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
		Type type = Type.kernel;
		switch (typeOrdinal) {
		case 0:
			type = Type.kernel;
			break;
		case 1:
			type = Type.network;
			break;
		default:
			type = Type.kernel;
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
	 * Converting kernel type to integer number.
	 * @param kernelType kernel type.
	 * @return integer number.
	 */
	public static int kernelTypeToInt(KernelType kernelType) {
		return kernelType.ordinal();
	}

	
	/**
	 * Converting integer number to kernel type.
	 * @param kernelTypeOrdinal integer number.
	 * @return kernel type.
	 */
	public static KernelType intToKernelType(int kernelTypeOrdinal) {
		KernelType kernelType = KernelType.normal;
		switch (kernelTypeOrdinal) {
		case 0:
			kernelType = KernelType.normal;
			break;
		case 1:
			kernelType = KernelType.transformer;
			break;
		case 2:
			kernelType = KernelType.filter_activate;
			break;
		case 3:
			kernelType = KernelType.weight_activate;
			break;
		case 4:
			kernelType = KernelType.nil;
			break;
		default:
			kernelType = KernelType.normal;
			break;
		}
		return kernelType;
	}
	
	
	/**
	 * Converting string to kernel type.
	 * @param kernelTypeText string.
	 * @return kernel type.
	 */
	public static KernelType stringToKernelType(String kernelTypeText) {
		return KernelType.valueOf(kernelTypeText);
	}

	
	/**
	 * Converting network type to integer number.
	 * @param networkType network type.
	 * @return integer number.
	 */
	public static int networkTypeToInt(NetworkType networkType) {
		return networkType.ordinal();
	}
	

	/**
	 * Converting integer number to network type.
	 * @param networkTypeOrdinal integer number.
	 * @return type.
	 */
	public static NetworkType intToNetworkType(int networkTypeOrdinal) {
		NetworkType networkType = NetworkType.basic;
		switch (networkTypeOrdinal) {
		case 0:
			networkType = NetworkType.basic;
			break;
		default:
			networkType = NetworkType.basic;
			break;
		}
		return networkType;
	}
	
	
	/**
	 * Converting string to network type.
	 * @param networkTypeText string.
	 * @return network type.
	 */
	public static NetworkType stringToNetworkType(String networkTypeText) {
		return NetworkType.valueOf(networkTypeText);
	}

	
	/**
	 * Creating weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param hint hinting value.
	 * @param layerSpec layer specification, which can be null.
	 * @param neuronChannel neuron channel which is only applied to network weight.
	 * @return weight.
	 */
	public static Weight newWeight(Size prevSize, Size size, NeuronValue hint, LayerSpec layerSpec, int neuronChannel) {
		if (prevSize == null && size == null && layerSpec == null) return new NullWeight();
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		hint = hint != null ? hint : NeuronValueCreator.newNeuronValue(neuronChannel);
		if (prevSize != null && size != null && (layerSpec == null || layerSpec.weightSpec == null))
			return WeightImpl.create(prevSize, size, hint);
		
		if (prevSize == null || size == null || layerSpec == null || layerSpec.weightSpec == null) throw new IllegalArgumentException();
		
		Weight weight = null;
		switch (layerSpec.weightSpec.type) {
		case kernel:
			switch (layerSpec.weightSpec.kernelType) {
			case normal:
				weight = WeightImpl.create(prevSize, size, hint);
				break;
			case transformer:
				weight = TransformerWeight.create(neuronChannel, prevSize, size);
				break;
			case filter_activate:
				weight = new ActivateFWeight();
				break;
			case weight_activate:
				weight = new ActivateWWeight();
				break;
			case nil:
				weight = new NullWeight();
				break;
			default:
				weight = WeightImpl.create(prevSize, size, hint);
				break;
			}
			break;
		case network:
			weight = WeightNetworkImpl.create(prevSize, size, Math.max(layerSpec.fifthLength, 1), layerSpec.weightSpec.kernelType, neuronChannel);
			break;
		default:
			weight = WeightImpl.create(prevSize, size, hint);
			break;
		}
		return weight;
	}


	/**
	 * Creating weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param hint hinting value.
	 * @param layerSpec layer specification, which can be null.
	 * @return weight.
	 */
	static Weight newWeight(Size prevSize, Size size, NeuronValue hint, LayerSpec layerSpec) {
		return newWeight(prevSize, size, hint, layerSpec, 0);
	}
	
	
	/**
	 * Creating weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param layerSpec layer specification, which can be null.
	 * @param neuronChannel neuron channel which is only applied to network weight.
	 * @return weight.
	 */
	static Weight newWeight(Size prevSize, Size size, LayerSpec layerSpec, int neuronChannel) {
		return newWeight(prevSize, size, null, layerSpec, neuronChannel);
	}
	
	
	/**
	 * Creating normal weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param hint hinting value.
	 * @return normal weight.
	 */
	static Weight newWeight(Size prevSize, Size size, NeuronValue hint) {
		return newWeight(prevSize, size, hint, (LayerSpec)null, 0);
	}
	
	
	/**
	 * Creating weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param neuronChannel neuron channel which is only applied to network weight.
	 * @return weight.
	 */
	static Weight newWeight(Size prevSize, Size size, int neuronChannel) {
		return newWeight(prevSize, size, null, (LayerSpec)null, neuronChannel);
	}
	
	
	/**
	 * Creating null weight.
	 * @return null weight.
	 */
	static Weight newWeight() {return new NullWeight();}


	/**
	 * Creating weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param hint hinting value.
	 * @param weightSpec weight specification, which can be null.
	 * @param neuronChannel neuron channel which is only applied to network weight.
	 * @return weight.
	 */
	public static Weight newWeight(Size prevSize, Size size, NeuronValue hint, WeightSpec weightSpec, int neuronChannel) {
		if (prevSize == null && size == null && weightSpec == null) return new NullWeight();
		if (prevSize != null && size != null && weightSpec == null)
			return WeightImpl.create(prevSize, size, hint);
		
		if (prevSize == null || size == null || weightSpec == null) throw new IllegalArgumentException();
		
		Weight weight = null;
		switch (weightSpec.type) {
		case kernel:
			switch (weightSpec.kernelType) {
			case normal:
				weight = WeightImpl.create(prevSize, size, hint);
				break;
			case transformer:
				weight = TransformerWeight.create(neuronChannel, prevSize, size);
				break;
			case filter_activate:
				weight = new ActivateFWeight();
				break;
			case weight_activate:
				weight = new ActivateWWeight();
				break;
			case nil:
				weight = new NullWeight();
				break;
			default:
				weight = WeightImpl.create(prevSize, size, hint);
				break;
			}
			break;
		case network:
			throw new IllegalArgumentException();
		default:
			weight = WeightImpl.create(prevSize, size, hint);
			break;
		}
		return weight;
	}


	/**
	 * Creating weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param hint hinting value.
	 * @param weightSpec weight specification, which can be null.
	 * @param neuronChannel neuron channel which is only applied to network weight.
	 * @return weight.
	 */
	static Weight newWeight(Size prevSize, Size size, NeuronValue hint, WeightSpec weightSpec) {
		return newWeight(prevSize, size, hint, weightSpec, 0);
	}
	
	
	/**
	 * Creating weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param weightSpec weight specification, which can be null.
	 * @param neuronChannel neuron channel which is only applied to network weight.
	 * @return weight.
	 */
	static Weight newWeight(Size prevSize, Size size, WeightSpec weightSpec, int neuronChannel) {
		return newWeight(prevSize, size, null, weightSpec, neuronChannel);
	}
	
	
}
