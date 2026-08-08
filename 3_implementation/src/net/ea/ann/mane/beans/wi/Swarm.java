/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans.wi;

import java.util.List;
import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.ParameterLayer;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.beans.wi.Swarm.Particle;
import net.ea.ann.raster.Raster;

/**
 * This class represents swarm in PSO matrix neural network.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Swarm extends VGGExt {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for number of particles.
	 */
	public final static String PARTICLES_COUNT_FIELD = "swarm_particles";
	
	
	/**
	 * Default value for number of particles.
	 */
	public final static int PARTICLES_COUNT_DEFAULT = 0;

	
	/**
	 * This interface represents particle.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static interface Particle {
		
		/**
		 * Initializing particle.
		 * @param layerSpecs array of layer specifications.
		 * @param dual dual mode.
		 * @return true if initialization is successful.
		 */
		boolean initialize(net.ea.ann.mane.MatrixLayerAbstract.LayerSpec[] layerSpecs, boolean dual);

		/**
		 * Learning particle.
		 * @param sample sample.
		 * @param learningRate learning rate.
		 * @return
		 */
		Error[] learn(Iterable<Record> sample, double learningRate);

		/**
		 * Learning particle.
		 * @param sample sample.
		 * @param learningRate learning rate.
		 * @return
		 */
		Error[] learnRaster(Iterable<Raster> sample, double learningRate);

		/**
		 * Getting current position.
		 * @return current position.
		 */
		NetworkParameter getPosition();
		
		/**
		 * Cloning current position.
		 * @return cloned current position.
		 */
		NetworkParameter clonePosition();

		/**
		 * Setting current position.
		 * @param position current position.
		 */
		void setPosition(NetworkParameter position);
		
		
		/**
		 * Getting current velocity.
		 * @return current velocity.
		 */
		NetworkParameter getVelocity();
		
		
		/**
		 * Setting current velocity.
		 * @param velocity current velocity.
		 */
		void setVelocity(NetworkParameter velocity);
		
		
		/**
		 * Getting local best position.
		 * @return local best position.
		 */
		NetworkParameter getBestPosition();

		/**
		 * Setting best position.
		 * @param bestPosition best position.
		 */
		void setBestPosition(NetworkParameter bestPosition);
		
		/**
		 * Getting best position.
		 * @return best position.
		 */
		NetworkParameter getGlobalBestPosition();
	
		/**
		 * Getting target value.
		 * @return target value.
		 */
		double target();

	}
	
	
	/**
	 * This functional interface represents swarm learner.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	@FunctionalInterface
	static interface Learner {
		
		/**
		 * Learning sample.
		 * @param particle particle.
		 * @return learning errors.
		 */
		Error[] learn(Particle particle);
		
	}
	
	
	/**
	 * List of particles which are particular networks.
	 */
	protected List<Particle> particles = Util.newList(0);
	
	
	/**
	 * Global best position.
	 */
	protected NetworkParameter globalBestPosition = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public Swarm(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		this.config.put(PARTICLES_COUNT_FIELD, PARTICLES_COUNT_DEFAULT);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public Swarm(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public Swarm(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public Swarm(int neuronChannel) {this(neuronChannel, null, null, null);}

	
	@Override
	MatrixLayerAbstract newNormalLayer(MatrixLayerAbstract.LayerSpec layerSpec) {
		ParameterLayer layer = new ParameterLayer(neuronChannel, getActivateRef(), getConvActivateRef(), idRef);
		layer.setNetwork(this);
		return layer;
	}


	/**
	 * Creating particle.
	 * @return particle.
	 */
	protected Particle createParticle() {
		return new ParticleImpl(this.neuronChannel, this.activateRef, this.convActivateRef, this.idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public NetworkParameter getGlobalBestPosition() {
				Swarm thisSwarm = thisSwarm();
				return thisSwarm.globalBestPosition;
			}
			
		};
	}
	
	
	/**
	 * Creating randomizer.
	 * @return randomizer.
	 */
	protected Randomizer createRandomizer() {
		final Random rnd0 = new Random();
		return new Randomizer() {
			@Override
			public double rand() {return rnd0.nextDouble();}
		};
	}
	
	
	/**
	 * Getting this swarm.
	 * @return this swarm.
	 */
	Swarm thisSwarm() {return this;}
	
	
	/**
	 * Getting number of individual particles.
	 * @return number of individual particles.
	 */
	public int countParticles() {return this.particles.size();}
	
	
	/**
	 * Getting individual particle.
	 * @param index index.
	 * @return individual particle.
	 */
	public Particle particle(int index) {return this.particles.get(index);}


	@Override
	public void reset() {
		super.reset();
		this.particles.clear();
		this.globalBestPosition = null;
	}


	@Override
	protected boolean initialize(net.ea.ann.mane.MatrixLayerAbstract.LayerSpec[] layerSpecs, boolean dual) {
		if (!super.initialize(layerSpecs, dual)) return false;
		this.particles.clear();
		this.globalBestPosition = null;
		
		int particlesCount = paramGetParticlesCount();
		if (particlesCount > 1) {
			for (int i = 0; i < particlesCount; i++) {
				Particle particle = createParticle();
				if (particle instanceof MatrixNetworkImpl) {
					try {
						((MatrixNetworkImpl)particle).setConfig(getConfig());
					} catch (Throwable e) {Util.trace(e);}
				}
				
				if (!particle.initialize(layerSpecs, dual)) return false;
				this.particles.add(particle);
			}
		}
		
		return true;
	}


	/**
	 * Learning swarm.
	 * @param <T> record type.
	 * @param sample sample.
	 * @param learningRate learning rate.
	 * @param learner learner.
	 * @return learning errors.
	 */
	<T> Error[] learn0(Iterable<T> sample, double learningRate, Learner learner) {
		NetworkParameter globalBest = null;
		try {
			if (this.globalBestPosition != null) globalBest = (NetworkParameter)((CloneableParameter)this.globalBestPosition).clone();
		} catch (Throwable e) {Util.trace(e);}
		Randomizer rnd = createRandomizer();

		//Learning particular particles.
		int particlesCount = countParticles();
		Error[][] errorsList = new Error[particlesCount][];
		double minTarget = Float.MAX_VALUE;
		int minIndex = -1;
		for (int i = 0; i < particlesCount; i++) {
			//Updating velocity and position.
			if (particle(i).getBestPosition() != null && this.globalBestPosition != null) {
				NetworkParameter position = null, localBest = null;
				try {
					position = (NetworkParameter)((CloneableParameter)particle(i).getPosition()).clone();
					localBest = (NetworkParameter)((CloneableParameter)particle(i).getBestPosition()).clone();
				} catch (Throwable e) {Util.trace(e);}
				
				NetworkParameter localDev = (NetworkParameter)localBest.psubtract(position).
					pmultiplyRandom(rnd);
				NetworkParameter globalDev = (NetworkParameter)globalBest.psubtract(position).
					pmultiplyRandom(rnd);
				
				NetworkParameter velocity = (NetworkParameter)particle(i).getVelocity().padd(localDev.padd(globalDev));
				particle(i).setVelocity(velocity);
				particle(i).setPosition((NetworkParameter)position.padd(velocity));
			}
			
			//Learning particle.
			double oldTarget = particle(i).target();
			Error[] errors = learner.learn(particle(i));
			errorsList[i] = errors;

			//Updating best position.
			double target = particle(i).target();
			if (particle(i).getBestPosition() == null || target < oldTarget) {
				particle(i).setBestPosition(particle(i).getPosition());
			}
			
			if (target < minTarget) {
				minIndex = i;
				minTarget = target;
			}
		}
		
		//Updating global best position.
		this.globalBestPosition = particle(minIndex).getPosition();
		this.pcopy(globalBestPosition);
		
		return errorsList[minIndex];
	}
	
	@Override
	protected Error[] learn(Iterable<Record> sample, double learningRate) {
		if (paramGetParticlesCount() > 1) {
			return learn0(sample, learningRate, (Particle particle) -> {
				return particle.learn(sample, learningRate);
			});
		}
		else {
			return super.learn(sample, learningRate);
		}
	}
	
	
	@Override
	Error[] learnRaster(Iterable<Raster> sample, double learningRate) {
		if (paramGetParticlesCount() > 1) {
			return learn0(sample, learningRate, (Particle particle) -> {
				return particle.learnRaster(sample, learningRate);
			});
		}
		else {
			return super.learnRaster(sample, learningRate);
		}
	}


	/**
	 * Getting number of particles.
	 * @return number of particles.
	 */
	public int paramGetParticlesCount() {
		if (config.containsKey(PARTICLES_COUNT_FIELD))
			return config.getAsInt(PARTICLES_COUNT_FIELD);
		else
			return PARTICLES_COUNT_DEFAULT;
	}
	
	
	/**
	 * Setting number of particles.
	 * @param particles number of particles.
	 * @return this swarm.
	 */
	public Swarm paramSetParticlesCount(int particles) {
		particles = particles < 1 ? PARTICLES_COUNT_DEFAULT : particles;
		config.put(PARTICLES_COUNT_FIELD, particles);
		return this;
	}
	
	
}



/**
 * This class represents particle in swarm.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class ParticleImpl extends VGGExt implements Particle {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal velocity.
	 */
	protected NetworkParameter velocity = null;
	
	
	/**
	 * Best local position.
	 */
	protected NetworkParameter bestPosition = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ParticleImpl(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ParticleImpl(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ParticleImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	@Override
	MatrixLayerAbstract newNormalLayer(MatrixLayerAbstract.LayerSpec layerSpec) {
		ParameterLayer layer = new ParameterLayer(neuronChannel, getActivateRef(), getConvActivateRef(), idRef);
		layer.setNetwork(this);
		return layer;
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ParticleImpl(int neuronChannel) {this(neuronChannel, null, null, null);}

	
	@Override
	public NetworkParameter getPosition() {return this.extractParameter();}

	
	@Override
	public NetworkParameter clonePosition() {return this.cloneParameter();}

	
	@Override
	public void setPosition(NetworkParameter position) {this.pcopy(position);}
	
	
	@Override
	public NetworkParameter getVelocity() {return this.velocity;}
	
	
	@Override
	public void setVelocity(NetworkParameter velocity) {this.velocity = velocity;}
	
	
	@Override
	public NetworkParameter getBestPosition() {return this.bestPosition;}


	@Override
	public void setBestPosition(NetworkParameter bestPosition) {this.bestPosition = bestPosition;}
	
	
	@Override
	public void reset() {
		super.reset();
		this.velocity = null;
		this.bestPosition = null;
	}


	@Override
	public double target() {
		return MatrixUtil.valueSum(getInputLayer().getBias()).mean();
	}
	
	
	@Override
	public boolean initialize(net.ea.ann.mane.MatrixLayerAbstract.LayerSpec[] layerSpecs, boolean dual) {
		if (!super.initialize(layerSpecs, dual)) return false;
		
		this.velocity = (NetworkParameter)cloneParameter().pinit(() -> {
			return new Random().nextDouble();
		});
		this.bestPosition = null;
		return true;
	}


	@Override
	public Error[] learn(Iterable<Record> sample, double learningRate) {
		return super.learn(sample, learningRate);
	}


	@Override
	public Error[] learnRaster(Iterable<Raster> sample, double learningRate) {
		return super.learnRaster(sample, learningRate);
	}


}
