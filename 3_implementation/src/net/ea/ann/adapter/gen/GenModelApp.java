/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.Comparator;

import javax.swing.JComboBox;
import javax.swing.JOptionPane;

import net.ea.ann.classifier.ClassifierAssoc;
import net.ea.ann.conv.ConvNetworkImpl;
import net.ea.ann.core.Util;
import net.ea.ann.gen.ConvGenModelAssoc;
import net.ea.ann.mane.MatrixNetworkAssoc;
import net.hudup.core.client.ConnectInfo;
import net.hudup.core.client.PowerServer;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.console.Console;
import net.hudup.core.logistic.console.ConsoleApp;
import net.hudup.core.logistic.console.ConsoleCP;
import net.hudup.core.logistic.console.ConsoleImpl;
import net.hudup.core.logistic.ui.StartDlg;
import net.hudup.core.logistic.ui.TextArea;

/**
 * This class represents generative model application.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GenModelApp extends ConsoleApp {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Name of generative AI task.
	 */
	protected final static String GENAI = "genai";
	
	
	/**
	 * Name of generative AI task.
	 */
	protected final static String LEARN_FILTER = "learn_filter";

	
	/**
	 * Name of transformation task.
	 */
	protected final static String MANE_TRANSFORM = "mane_transform";

	
	/**
	 * Name of classification task.
	 */
	protected final static String CLASSIFY = "classification";

	
	/**
	 * Array of task.
	 */
	protected final static String[] tasks = {GENAI, LEARN_FILTER, MANE_TRANSFORM, CLASSIFY};
	
	
	/**
	 * Internal task.
	 */
	protected String task = tasks[0];
	
	
	/**
	 * Constructor with server, generative model application creator, and console.
	 * @param server power server.
	 * @param gmAppor generative model application creator.
	 * @param console console.
	 */
	public GenModelApp(PowerServer server, GenModelAppor gmAppor, Console console) {
		super(server, gmAppor, console);
	}

	
	@Override
	public String getDesc() throws RemoteException {
		return "GenAI tester";
	}


	@Override
	protected void consoleTask() {
		switch (task) {
		case GENAI:
			try {
				ConvGenModelAssoc.gen(((ConsoleImpl)console).getIn(), ((ConsoleImpl)console).getOut());
			} catch (Throwable e) {Util.trace(e);}
			break;
		case LEARN_FILTER:
			try {
				ConvNetworkImpl.learnFilter(((ConsoleImpl)console).getIn(), ((ConsoleImpl)console).getOut());
			} catch (Throwable e) {Util.trace(e);}
			break;
		case MANE_TRANSFORM:
			try {
				MatrixNetworkAssoc.transform(((ConsoleImpl)console).getIn(), ((ConsoleImpl)console).getOut());
			} catch (Throwable e) {Util.trace(e);}
			break;
		case CLASSIFY:
			try {
				ClassifierAssoc.classify(((ConsoleImpl)console).getIn(), ((ConsoleImpl)console).getOut());
			} catch (Throwable e) {Util.trace(e);}
			break;
		default:
			try {
				ConvGenModelAssoc.gen(((ConsoleImpl)console).getIn(), ((ConsoleImpl)console).getOut());
			} catch (Throwable e) {Util.trace(e);}
		}
	}

	
	@Override
	public void show(ConnectInfo connectInfo) throws RemoteException {
		if (connectInfo == null) return;
		try {
			ConsoleCP ccp = new ConsoleCP(console, connectInfo) {

				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				protected void changeTask() {
					changeConsoleTask(this);
				}
				
			};
			if ((ccp.getConnectInfo().bindUri == null) && (console instanceof ConsoleImpl))
				ccp.setTitle(getDesc() + " - " + task);
			else
				ccp.setTitle(getDesc());
			ccp.setVisible(true);
		} catch (Throwable e) {LogUtil.trace(e);}
	}


	/**
	 * Changing task.
	 * @param ccp console control panel.
	 */
	protected void changeConsoleTask(ConsoleCP ccp) {
		if ((ccp.getConnectInfo().bindUri != null) || !(console instanceof ConsoleImpl)) {
			JOptionPane.showMessageDialog(ccp, "Unable to change remotely task", "Unable to change remotely task", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		boolean started = false;
		try {
			started = console.isConsoleStarted();
		} catch (Throwable e) {LogUtil.trace(e);}
		if (started) {
			JOptionPane.showMessageDialog(ccp, "Unable to change task because some task was started", "Unable to change task", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		Arrays.sort(tasks, new Comparator<String>() {

			@Override
			public int compare(String o1, String o2) {
				return o1.compareTo(o2);
			}
			
		});

		GenModelApp thisApp = this;
		final StartDlg dlgStarter = new StartDlg(ccp, "Tasks") {
			
			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected void start() {
				String task = (String)getItemControl().getSelectedItem();
				dispose();
				if (task == null) return;
				
				thisApp.task = task;
				try {
					ccp.setTitle(thisApp.getDesc() + " - " + task);
				} catch (Throwable e) {Util.trace(e);}
				JOptionPane.showMessageDialog(this, "Task changed into \"" + task + "\".\nPlease run the new task.", "Task changed", JOptionPane.INFORMATION_MESSAGE);
			}
			
			@Override
			protected JComboBox<?> createItemControl() {
				return new JComboBox<String>(tasks);
			}
			
			@Override
			protected TextArea createHelp() {
				TextArea tooltip = new TextArea("Please select a task and then press \"Start\" button");
				tooltip.setEditable(false);
				return tooltip;
			}
			
		};

		dlgStarter.getItemControl().setSelectedItem(task);
		dlgStarter.setSize(400, 200);
		dlgStarter.setLocationRelativeTo(ccp);
        dlgStarter.setVisible(true);
	}


}
