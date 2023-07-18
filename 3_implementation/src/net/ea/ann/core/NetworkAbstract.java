/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.rmi.NoSuchObjectException;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

/**
 * This class is basic abstract implementation of neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class NetworkAbstract implements Network {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Maximum iteration of learning neural network.
	 */
	public final static String LEARN_MAX_ITERATION_FIELD = "learn_max_iteration";
	
	
	/**
	 * Terminated threshold of learning neural network.
	 */
	public final static String LEARN_TERMINATED_THRESHOLD_FIELD = "learn_terminated_threshold";

	
	/**
	 * Learning rate.
	 */
	public final static String LEARN_RATE_FIELD = "learn_rate";

	
	/**
	 * Holding a list of listeners.
	 */
    protected transient NetworkListenerList listenerList = new NetworkListenerList();

    
    /**
     * Flag to indicate whether algorithm learning process was started.
     */
    protected volatile boolean doStarted = false;
    
    
    /**
     * Flag to indicate whether algorithm learning process was paused.
     */
    protected volatile boolean doPaused = false;

    
	/**
	 * Configuration.
	 */
	protected NetworkConfig config = new NetworkConfig();
	
	
	/**
	 * Flag to indicate whether this hidden Markov model was exported.
	 */
	protected boolean exported = false;

	
    /**
	 * Internal identifier.
	 */
	protected Id idRef = new Id();
	
	
	/**
	 * Constructor with ID reference.
	 */
	public NetworkAbstract(Id idRef) {
		config.put(LEARN_MAX_ITERATION_FIELD, LEARN_MAX_ITERATION_DEFAULT);
		config.put(LEARN_TERMINATED_THRESHOLD_FIELD, LEARN_TERMINATED_THRESHOLD_DEFAULT);
		config.put(LEARN_RATE_FIELD, LEARN_RATE_DEFAULT);

		if (idRef != null) this.idRef = idRef;
	}

	/**
	 * Default constructor.
	 */
	public NetworkAbstract() {
		this(new Id());
	}

	
	@Override
	public void addListener(NetworkListener listener) throws RemoteException {
		synchronized (listenerList) {
			listenerList.add(NetworkListener.class, listener);
		}
	}


	@Override
	public void removeListener(NetworkListener listener) throws RemoteException {
		synchronized (listenerList) {
			listenerList.remove(NetworkListener.class, listener);
		}
	}
	
	
	/**
	 * Getting an array of listeners.
	 * @return array of listeners.
	 */
	protected NetworkListener[] getListeners() {
		if (listenerList == null) return new NetworkListener[] {};
		synchronized (listenerList) {
			return listenerList.getListeners(NetworkListener.class);
		}

	}
	
	
	/**
	 * Firing information event.
	 * @param evt information event.
	 */
	protected void fireInfoEvent(NetworkInfoEvent evt) {
		if (listenerList == null) return;
		
		NetworkListener[] listeners = getListeners();
		for (NetworkListener listener : listeners) {
			try {
				listener.receivedInfo(evt);
			}
			catch (Throwable e) { 
				Util.trace(e);
			}
		}
	}

	
	/**
	 * Firing learning event.
	 * @param evt learning event.
	 */
	protected void fireDoEvent(NetworkDoEvent evt) {
		if (listenerList == null) return;
		
		NetworkListener[] listeners = getListeners();
		for (NetworkListener listener : listeners) {
			try {
				listener.receivedDo(evt);
			}
			catch (Throwable e) {
				Util.trace(e);
			}
		}
	}


	@Override
	public boolean doPause() throws RemoteException {
		if (!isDoRunning()) return false;
		
		doPaused  = true;
		
		try {
			wait();
		} 
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return true;
	}


	@Override
	public boolean doResume() throws RemoteException {
		if (!isDoPaused()) return false;
		
		doPaused = false;
		notifyAll();
		
		return true;
	}


	@Override
	public boolean doStop() throws RemoteException {
		if (!isDoStarted()) return false;
		
		doStarted = false;
		
		if (doPaused) {
			doPaused = false;
			notifyAll();
		}
		
		try {
			wait();
		} 
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return true;
	}


	@Override
	public boolean isDoStarted() throws RemoteException {
		return doStarted;
	}


	@Override
	public boolean isDoPaused() throws RemoteException {
		return doStarted && doPaused;
	}


	@Override
	public boolean isDoRunning() throws RemoteException {
		return doStarted && !doPaused;
	}

	
	@Override
	public NetworkConfig getConfig() throws RemoteException {
		return config;
	}


	@Override
	public void setConfig(NetworkConfig config) throws RemoteException {
		if (config != null) this.config.putAll(config);
	}


	@Override
	public synchronized Remote export(int serverPort) throws RemoteException {
		if (exported) return null;
		
		Remote stub = null;
		try {
			stub = UnicastRemoteObject.exportObject(this, serverPort);
		}
		catch (Exception e) {
			try {
				if (stub != null) UnicastRemoteObject.unexportObject(this, true);
			}
			catch (Exception e2) {}
			stub = null;
		}
	
		exported = stub != null;
		return stub;
	}


	@Override
	public synchronized void unexport() throws RemoteException {
		if (!exported) return;

		try {
        	UnicastRemoteObject.unexportObject(this, true);
			exported = false;
		}
		catch (NoSuchObjectException e) {
			exported = false;
			Util.trace(e);
		}
		catch (Throwable e) {
			Util.trace(e);
		}
	}

	
	@Override
	public void close() throws Exception {
		try {
			unexport();
		}
		catch (Throwable e) {
			Util.trace(e);
		}
	}


}
