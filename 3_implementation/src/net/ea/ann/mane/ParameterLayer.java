/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.ParameterLayer.LayerParameter;

/**
 * This class implements parameter layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ParameterLayer extends MatrixLayerImpl implements Parameter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This interface represents layer parameter.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static interface LayerParameter extends Parameter {}

	
	/**
	 * This class represents null layer parameter.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class NullLayerParameter extends NullParameter implements LayerParameter {

		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public NullLayerParameter() {
			super();
		}
		
	}

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ParameterLayer(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ParameterLayer(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ParameterLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ParameterLayer(int neuronChannel) {this(neuronChannel, null, null, null);}


	/**
	 * Extracting parameter.
	 * @return extracted parameter.
	 */
	protected LayerParameter extractParameter() {
		return new LayerParameterImpl(getWeight(), Kernel.GLOBAL_BIAS?getBias():null, getFilter(), Kernel.GLOBAL_BIAS?getFilterBias():null);
	}
	
	
	/**
	 * Cloning parameter.
	 * @return cloned parameter.
	 */
	LayerParameter cloneParameter() {
		try {
			return (LayerParameter)((CloneableParameter)extractParameter()).clone();
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
	@Override
	public Parameter pcopy(Parameter other) {
		if (this == other) return this;
		LayerParameterImpl player = (LayerParameterImpl)other;
		if (this.weight != null && player.getWeight() != null && this.weight != player.getWeight()) {
			this.weight.pcopy(player.getWeight());
		}
		if (this.bias != null && player.getBias() != null && this.bias != player.getBias()) {
			MatrixUtil.copy(player.getBias(), this.bias);
		}
		if (this.filter != null && player.getFilter() != null && this.filter != player.getFilter()) {
			this.filter.pcopy(player.getFilter());
		}
		if (this.filterBias != null && player.getFilterBias() != null && this.filterBias != player.getFilterBias()) {
			this.filterBias = player.getFilterBias();
		}
		return this;
	}


	@Override
	public Parameter padd(Parameter other) {
		pcopy(extractParameter().padd(other));
		return this;
	}


	@Override
	public Parameter psubtract(Parameter other) {
		pcopy(extractParameter().psubtract(other));
		return this;
	}


	@Override
	public Parameter pmultiply(double factor) {
		pcopy(extractParameter().pmultiply(factor));
		return this;
	}


	@Override
	public Parameter pmultiplyRandom(Randomizer rnd) {
		pcopy(extractParameter().pmultiplyRandom(rnd));
		return this;
	}


	@Override
	public Parameter pinit(Randomizer rnd) {
		pcopy(extractParameter().pinit(rnd));
		return this;
	}


}


/**
 * This class implements layer parameter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class LayerParameterImpl implements LayerParameter, Parameter.CloneableParameter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Parametric weight.
	 */
	protected Weight weight = null;
	
	
	/**
	 * Bias.
	 */
	protected Matrix bias = null;

	
	/**
	 * Convolutional filter.
	 */
	protected Filter filter = null;
	
	
	/**
	 * Convolutional filter bias.
	 */
	protected NeuronValue filterBias = null;


	/**
	 * Default constructor.
	 */
	public LayerParameterImpl() {}

	
	/**
	 * Constructor with weight, bias, filter, and filter bias.
	 * @param weight weight.
	 * @param bias bias.
	 * @param filter filter.
	 * @param filterBias filter bias.
	 */
	public LayerParameterImpl(Weight weight, Matrix bias, Filter filter, NeuronValue filterBias) {
		this.weight = weight;
		this.bias = bias;
		this.filter = filter;
		this.filterBias = filterBias;
	}
	
	
	/**
	 * Constructor with weight and filter.
	 * @param weight weight.
	 * @param filter filter.
	 */
	public LayerParameterImpl(Weight weight, Filter filter) {
		this(weight, null, filter, null);
	}
	
	
	/**
	 * Getting weight.
	 * @return the weight.
	 */
	public Weight getWeight() {return weight;}

	
	/**
	 * Getting bias.
	 * @return bias.
	 */
	public Matrix getBias() {return bias;}


	/**
	 * Getting convolutional filter.
	 * @return convolutional filter.
	 */
	public Filter getFilter() {return filter;}


	/**
	 * Getting convolutional filter bias.
	 * @return convolutional filter bias.
	 */
	public NeuronValue getFilterBias() {return filterBias;}


	@Override
	public Parameter pcopy(Parameter other) {
		if (this == other) return this;
		LayerParameterImpl player = (LayerParameterImpl)other;
		if (this.weight != null && player.getWeight() != null && this.weight != player.getWeight()) {
			this.weight.pcopy(player.getWeight());
		}
		if (this.bias != null && player.getBias() != null && this.bias != player.getBias()) {
			MatrixUtil.copy(player.getBias(), this.bias);
		}
		if (this.filter != null && player.getFilter() != null && this.filter != player.getFilter()) {
			this.filter.pcopy(player.getFilter());
		}
		if (this.filterBias != null && player.getFilterBias() != null && this.filterBias != player.getFilterBias()) {
			this.filterBias = player.getFilterBias();
		}
		
		return this;
	}


	@Override
	public Parameter padd(Parameter other) {
		LayerParameterImpl player = (LayerParameterImpl)other;
		if (this.weight != null && player.getWeight() != null) {
			this.weight = (Weight)this.getWeight().padd(player.getWeight());
		}
		if (this.bias != null && player.getBias() != null) {
			this.bias = this.getBias().add(player.getBias());
		}
		if (this.filter != null && player.getFilter() != null) {
			this.filter = (Filter)this.getFilter().padd(player.getFilter());
		}
		if (this.filterBias != null && player.getFilterBias() != null) {
			this.filterBias = this.getFilterBias().add(player.getFilterBias());
		}
		
		return this;
	}


	@Override
	public Parameter psubtract(Parameter other) {
		LayerParameterImpl player = (LayerParameterImpl)other;
		if (this.weight != null && player.getWeight() != null) {
			this.weight = (Weight)this.getWeight().psubtract(player.getWeight());
		}
		if (this.bias != null && player.getBias() != null) {
			this.bias = this.getBias().subtract(player.getBias());
		}
		if (this.filter != null && player.getFilter() != null) {
			this.filter = (Filter)this.getFilter().psubtract(player.getFilter());
		}
		if (this.filterBias != null && player.getFilterBias() != null) {
			this.filterBias = this.getFilterBias().subtract(player.getFilterBias());
		}
		
		return this;
	}


	@Override
	public Parameter pmultiply(double factor) {
		if (this.weight != null) {
			this.weight = (Weight)this.getWeight().pmultiply(factor);
		}
		if (this.bias != null) {
			this.bias = this.getBias().multiply0(factor);
		}
		if (this.filter != null) {
			this.filter = (Filter)this.getFilter().pmultiply(factor);
		}
		if (this.filterBias != null) {
			this.filterBias = this.getFilterBias().multiply(factor);
		}
		
		return this;
	}


	@Override
	public Parameter pmultiplyRandom(Randomizer rnd) {
		if (this.weight != null) this.weight.pmultiplyRandom(rnd);
		if (this.bias != null) MatrixUtil.fillMulti(this.bias, rnd);
		
		if (this.filter != null) this.filter.pmultiplyRandom(rnd);
		if (this.filterBias != null) this.filterBias = this.filterBias.multiply(rnd.rand());
		
		return this;
	}


	@Override
	public Parameter pinit(Randomizer rnd) {
		if (this.weight != null) this.weight.pinit(rnd);
		if (this.bias != null) MatrixUtil.fill(this.bias, 0);
		
		if (this.filter != null) this.filter.pinit(rnd);
		if (this.filterBias != null) this.filterBias = this.filterBias.zero();
		
		return this;
	}


	@Override
	public Object clone() throws CloneNotSupportedException {
		Weight clonedWeight = null;
		if (this.weight != null && this.weight instanceof Parameter.CloneableParameter) {
			clonedWeight = (Weight)((Parameter.CloneableParameter)this.weight).clone();
		}
		
		Matrix clonedBias = null;
		if (this.bias != null) {
			clonedBias = this.bias.create();
			MatrixUtil.copy(this.bias, clonedBias);
		}

		Filter clonedFilter = null;
		if (this.filter != null && this.filter instanceof Parameter.CloneableParameter) {
			clonedFilter = (Filter)((Parameter.CloneableParameter)this.filter).clone();
		}

		NeuronValue clonedFilterBias = this.filterBias;
		
		LayerParameterImpl cloned = new LayerParameterImpl(clonedWeight, clonedBias, clonedFilter, clonedFilterBias);
		return cloned;
	}


}


