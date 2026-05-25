/**
 * AI project provide artificial intelligence solutions
 * (C) Copyright by Loc Nguyen
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.hudup;

import java.util.Set;

import net.ea.ann.adapter.gen.GenModelRemote;
import net.ea.ann.adapter.gen.GenModelRemoteWrapper;
import net.ea.pso.adapter.PSORemote;
import net.ea.pso.adapter.PSORemoteWrapper;
import net.hudup.core.Firer;
import net.hudup.core.alg.AlgRemote;
import net.hudup.core.alg.AlgRemoteWrapper;

/**
 * This is advanced plug-in manager.
 * 
 * @author Loc Nguyen
 * @version 2.0
 *
 */
public class AIFirer extends AIFirer0 {

	
	@Override
	public void fireSimply() {
		super.fireSimply();
	}

	
	@Override
	public AlgRemoteWrapper wrap(AlgRemote remoteAlg, boolean exclusive) {
		if (remoteAlg instanceof GenModelRemote)
			return new GenModelRemoteWrapper((GenModelRemote)remoteAlg, exclusive);
		else if (remoteAlg instanceof PSORemote)
			return new PSORemoteWrapper((PSORemote)remoteAlg, exclusive);
		else
			return super.wrap(remoteAlg, exclusive);
	}

	
}



/**
 * This is core advanced plug-in manager which derives from {@link Firer}.
 * 
 * @author Loc Nguyen
 * @version 2.0
 *
 */
class AIFirer0 extends Firer {

	
	/**
	 * Default constructor.
	 */
	public AIFirer0() {
		super();
//		RELECTIONS_LIB = false;
	}

	
	/*
	 * Applying other finding class library like ClassGraph (<a href="https://github.com/classgraph/classgraph">https://github.com/classgraph/classgraph</a>).
	 */
	@Override
	protected <T> Set<Class<? extends T>> findClasses2(String[] prefixList, Class<T> referredClass) {
		return super.findClasses2(prefixList, referredClass);
	}
	
	
}



//class ModernScanner {
//	
//	
//	public static void main(String[] args) {
//		String packageName = "net.hudup";
//		
//		//1. Initialize the scan with options
//		try (ScanResult scanResult = new ClassGraph()
//				.enableAllInfo()             // Scans annotations, methods, fields, etc.
//				.acceptPackages(packageName) // Targets your specific package and its children
//				.scan()) {                   // Performs the scan efficiently
//			
//			//2. Fetch all classes found in that package
//			List<Class<?>> classes = scanResult.getAllClasses().loadClasses();
//			
//			System.out.println("Found " + classes.size() + " classes under package " + packageName);
//			for (Class<?> clazz : classes) {
//				System.out.println(clazz.getName());
//			}
//		}
//	}
//	
//    
//}
