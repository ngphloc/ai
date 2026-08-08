/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.mane.ParameterLayer.LayerParameter;
import net.ea.ann.mane.ParameterLayer.NullLayerParameter;
import net.ea.ann.mane.ParameterNetwork.NetworkParameter;

/**
 * This class represents custom matrix neural network.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ParameterNetwork extends MatrixNetworkImpl implements Parameter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This interface represents network parameter.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static interface NetworkParameter extends Parameter {
		
		/**
		 * Getting count of layer parameters.
		 * @return count of layer parameters.
		 */
		int countPLayers();
		
		/**
		 * Getting layer parameter at specified index.
		 * @param index specified index.
		 * @return layer parameter at specified index.
		 */
		LayerParameter getPLayer(int index);
		
		/**
		 * Setting layer parameter.
		 * @param index index.
		 * @param player layer parameter.
		 * @return this network parameter.
		 */
		NetworkParameter setPLayer(int index, LayerParameter player);
		
	}
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ParameterNetwork(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ParameterNetwork(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ParameterNetwork(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ParameterNetwork(int neuronChannel) {this(neuronChannel, null, null, null);}


	/**
	 * Getting layer parameter.
	 * @param index index.
	 * @return layer parameter.
	 */
	LayerParameter getPLayer(int index) {
		MatrixLayerAbstract player = this.get(index);
		if (player instanceof ParameterLayer)
			return ((ParameterLayer)player).extractParameter();
		else if (player instanceof LayerParameter)
			return (LayerParameter)player;
		else
			return new NullLayerParameter();
	}


	/**
	 * Cloning layer parameter.
	 * @param index index.
	 * @return cloned parameter.
	 */
	LayerParameter clonePlayer(int index) {
		MatrixLayerAbstract player = this.get(index);
		if (player instanceof ParameterLayer)
			return ((ParameterLayer)player).cloneParameter();
		else if (player instanceof LayerParameter) {
			if (player instanceof CloneableParameter) {
				try {
					return (LayerParameter)((CloneableParameter)player).clone();
				} catch (Throwable e) {Util.trace(e);}
				return null;
			}
			else
				throw new IllegalArgumentException();
		}
		else
			return new NullLayerParameter();
	}
	
	
	/**
	 * Getting layer parameter.
	 * @param index index.
	 * @param player layer parameter.
	 * @return this network.
	 */
	ParameterNetwork setPLayer(int index, LayerParameter player) {
		((ParameterLayer)get(index)).pcopy(player);
		return this;
	}

	
	/**
	 * Extracting network parameter.
	 * @return network parameter.
	 */
	protected NetworkParameter extractParameter() {
		NetworkParameterImpl pnetwork = new NetworkParameterImpl();
		for (int i = 0; i < size(); i++) {
			LayerParameter layer = getPLayer(i);
			assert (layer != null);
			pnetwork.addPLayer(layer);
		}
		return pnetwork;
	}
	
	
	/**
	 * Cloning network parameter.
	 * @return cloned network parameter.
	 */
	protected NetworkParameter cloneParameter() {
		try {
			return (NetworkParameter)((CloneableParameter)extractParameter()).clone();
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
	@Override
	public Parameter pcopy(Parameter other) {
		if (this == other) return this;
		NetworkParameter pnetwork = (NetworkParameter)other;
		if (pnetwork.countPLayers() != this.size()) throw new IllegalArgumentException();
		
		for (int i = 0; i < this.size(); i++) {
			if (this.getPLayer(i) != pnetwork.getPLayer(i)) this.getPLayer(i).pcopy(pnetwork.getPLayer(i));
		}
		return this;
	}


	@Override
	public Parameter padd(Parameter other) {
		pcopy(cloneParameter().padd(other));
		return this;
	}


	@Override
	public Parameter psubtract(Parameter other) {
		pcopy(cloneParameter().psubtract(other));
		return this;
	}


	@Override
	public Parameter pmultiply(double factor) {
		pcopy(cloneParameter().pmultiply(factor));
		return this;
	}

	
	@Override
	public Parameter pmultiplyRandom(Randomizer rnd) {
		pcopy(cloneParameter().pmultiplyRandom(rnd));
		return this;
	}


	@Override
	public Parameter pinit(Randomizer rnd) {
		pcopy(cloneParameter().pinit(rnd));
		return this;
	}


	@Override
	protected Error[] learn(Iterable<Record> sample, double learningRate) {
		return super.learn(sample, learningRate);
	}


}



/**
 * This class implements network parameter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class NetworkParameterImpl implements NetworkParameter, Parameter.CloneableParameter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of layer parameters.
	 */
	protected List<LayerParameter> players = Util.newList(0);
	
	
	/**
	 * Default constructor.
	 */
	public NetworkParameterImpl() {
		
	}


	@Override
	public int countPLayers() {return players.size();}


	@Override
	public LayerParameter getPLayer(int index) {return players.get(index);}
	
	
	/**
	 * Adding layer parameter.
	 * @param player layer parameter.
	 * @return true if adding is successful.
	 */
	boolean addPLayer(LayerParameter player) {return this.players.add(player);}
	
	
	/**
	 * Adding layer parameter.
	 * @param layer layer.
	 * @return true if adding is successful.
	 */
	boolean addLayer(MatrixLayerAbstract layer) {
		LayerParameter player = layer instanceof ParameterLayer ? ((ParameterLayer)layer).extractParameter() : new NullLayerParameter();
		return addPLayer(player);
	}
	
	
	/**
	 * Adding cloned layer parameter.
	 * @param layer layer.
	 * @return true if adding is successful.
	 */
	boolean addCloneLayer(MatrixLayerAbstract layer) {
		LayerParameter player = layer instanceof ParameterLayer ? ((ParameterLayer)layer).cloneParameter() : new NullLayerParameter();
		return addPLayer(player);
	}
	

	@Override
	public NetworkParameter setPLayer(int index, LayerParameter player) {
		this.players.set(index, player);
		return this;
	}


	/**
	 * Removing layer parameter.
	 * @param index index.
	 * @return removed layer parameter.
	 */
	LayerParameter removeLayer(int index) {return this.players.remove(index);}
	
	
	/**
	 * Clearing layer parameters.
	 */
	void clearLayers() {this.players.clear();}


	@Override
	public Parameter pcopy(Parameter other) {
		if (this == other) return this;
		NetworkParameter pnetwork = (NetworkParameter)other;
		if (pnetwork.countPLayers() != this.countPLayers()) throw new IllegalArgumentException();
		
		for (int i = 0; i < this.countPLayers(); i++) this.getPLayer(i).pcopy(pnetwork.getPLayer(i));
		return this;
	}


	@Override
	public Parameter padd(Parameter other) {
		NetworkParameter pnetwork = (NetworkParameter)other;
		if (pnetwork.countPLayers() != this.countPLayers()) throw new IllegalArgumentException();
		
		for (int i = 0; i < this.countPLayers(); i++) {
			LayerParameter layer = (LayerParameter)this.getPLayer(i).padd(pnetwork.getPLayer(i));
			if (layer != this.getPLayer(i)) throw new IllegalArgumentException();
		}
		return this;
	}


	@Override
	public Parameter psubtract(Parameter other) {
		NetworkParameter pnetwork = (NetworkParameter)other;
		if (pnetwork.countPLayers() != this.countPLayers()) throw new IllegalArgumentException();
		
		for (int i = 0; i < this.countPLayers(); i++) {
			LayerParameter layer = (LayerParameter)this.getPLayer(i).psubtract(pnetwork.getPLayer(i));
			if (layer != this.getPLayer(i)) throw new IllegalArgumentException();
		}
		return this;
	}


	@Override
	public Parameter pmultiply(double factor) {
		for (int i = 0; i < this.countPLayers(); i++) {
			LayerParameter player = (LayerParameter)this.getPLayer(i).pmultiply(factor);
			if (player != this.getPLayer(i)) throw new IllegalArgumentException();
		}
		return this;
	}


	@Override
	public Parameter pmultiplyRandom(Randomizer rnd) {
		for (int i = 0; i < this.countPLayers(); i++) {
			LayerParameter player = (LayerParameter)this.getPLayer(i).pmultiplyRandom(rnd);
			if (player != this.getPLayer(i)) throw new IllegalArgumentException();
		}
		return this;
	}
	
	
	@Override
	public Parameter pinit(Randomizer rnd) {
		for (int i = 0; i < this.countPLayers(); i++) {
			LayerParameter player = (LayerParameter)this.getPLayer(i).pinit(rnd);
			if (player != this.getPLayer(i)) throw new IllegalArgumentException();
		}
		return this;
	}


	@Override
	public Object clone() throws CloneNotSupportedException {
		NetworkParameterImpl cloned = new NetworkParameterImpl();
		for (int i = 0; i < this.players.size(); i++) {
			LayerParameter player = this.players.get(i);
			player = player instanceof NullParameter ? player : (LayerParameter) ((Parameter.CloneableParameter)player).clone();
			cloned.addPLayer(player);
		}
		return cloned;
	}


}
