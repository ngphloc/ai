/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import net.ea.ann.core.TextParsable;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.vector.NeuronValueVector;
import net.ea.ann.core.value.vector.WeightValueVectorImpl;

/**
 * This class represents connection weight between two neurons.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Weight implements TextParsable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Weight value.
	 */
	public WeightValue value = null;
	
	
	/**
	 * Constructor with real weight.
	 * @param value real weight.
	 */
	public Weight(WeightValue value) {
		this.value = value;
	}


	@Override
	public String toText() {
		if (value == null)
			return super.toString();
		else
			return "weight = " + value.toString();
	}


}



/**
 * This class represents connection weight between two neurons, associated with list of weight values.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
class MultiWeight extends Weight {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of weight values.
	 */
	protected List<WeightValue> wvs = Util.newList(0);
	
	
	/**
	 * Constructor with real weight.
	 * @param value real weight.
	 */
	public MultiWeight(WeightValue value) {
		super(value);
		wvs.add(value);
	}

	
	/**
	 * Constructor with array of values.
	 * @param values array of values.
	 */
	public MultiWeight(WeightValue...values) {
		super(values != null ? values[0] : null);
		wvs.addAll(Arrays.asList(values));
	}

	
	/**
	 * Constructor with collection of values.
	 * @param values collection of values.
	 */
	public MultiWeight(Collection<WeightValue> values) {
		super(null);
		wvs.addAll(values);
		for (WeightValue value : values) {
			if (value != null) {
				this.value = value;
				break;
			}
		}
	}

	
	/**
	 * Getting the number of values.
	 * @return  the number of values.
	 */
	public int size() {
		return wvs.size();
	}
	
	
	/**
	 * Getting value at specified index.
	 * @param index specified index.
	 * @return value at specified index.
	 */
	public WeightValue get(int index) {
		return wvs.get(index);
	}
	
	
	/**
	 * Setting value at specified index.
	 * @param index specified index.
	 * @param value specified value.
	 * @return previous value.
	 */
	public WeightValue set(int index, WeightValue value) {
		if (index == 0) this.value = value;
		return wvs.set(index, value);
	}
	
	
	/**
	 * Adding values.
	 * @param values specified values.
	 * @return this weight.
	 */
	public MultiWeight addValues(NeuronValue values) {
		if (values == null || wvs.size() == 0) return this;
		if (values instanceof NeuronValue1) {
			if (wvs.size() != 1) return this;
			WeightValue nw = wvs.get(0).addValue(values);
			wvs.set(0, nw);
		}
		else if (values instanceof NeuronValueV) {
			NeuronValueV vector = (NeuronValueV)values;
			int n = Math.min(vector.length(), wvs.size());
			for (int i = 0; i < n; i++) {
				WeightValue nw = wvs.get(i);
				nw = nw.addValue(nw.toValue().valueOf(vector.get(i)));
				wvs.set(i, nw);
			}
		}
		else if (values instanceof NeuronValueVector) {
			NeuronValueVector vector = (NeuronValueVector)values;
			int n = Math.min(vector.length(), wvs.size());
			for (int i = 0; i < n; i++) {
				WeightValue nw = wvs.get(i);
				nw = nw.addValue(vector.get(i));
				wvs.set(i, nw);
			}
		}
		
		if (wvs.size() > 0) this.value = wvs.get(0);
		return this;
	}
	
	
	/**
	 * Aggregating weight value.
	 * @return aggregated weight value.
	 */
	public WeightValue aggregateValues() {
		return aggregateValues(wvs);
	}
	
	
	/**
	 * Aggregating weight value.
	 * @return aggregated weight value.
	 */
	private static WeightValue aggregateValues(Collection<WeightValue> wvs) {
		if (wvs == null || wvs.size() == 0) return null;
		WeightValue value = null;
		for (WeightValue v : wvs) {
			if (v != null) {
				value = v;
				break;
			}
		}
		
		if (value == null || wvs.size() == 1)
			return value;
		else if (value instanceof WeightValue1) {
			List<Double> values = Util.newList(wvs.size());
			for (WeightValue v : wvs) values.add(((WeightValue1)v).get());
			return new WeightValueV(values);
		}
		else if (value instanceof WeightValueV) {
			List<WeightValue> values = Util.newList(wvs.size());
			for (WeightValue v : wvs) values.add(v);
			return new WeightValueVectorImpl(values);
		}
		else
			return null;
	}


}



