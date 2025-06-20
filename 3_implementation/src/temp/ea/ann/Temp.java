/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann;

import java.nio.file.Paths;
import java.util.List;

import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;

/**
 * This is temporary class.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Temp {

	
	/**
	 * Main method.
	 * @param args argument.
	 */
	public static void main(String[] args) {
//		try (NetworkStandardImpl network = new NetworkStandardImpl(1, new Logistic1())) {
//			network.initialize(3, 3, new int[] {3, 3, 3});
//			
//			Record record1 = new Record();
//			record1.input = new NeuronValue[] {new NeuronValue1(1), new NeuronValue1(2), new NeuronValue1(3)};
//			record1.output = new NeuronValue[] {new NeuronValue1(4), new NeuronValue1(5), new NeuronValue1(6)};
//
//			Record record2 = new Record();
//			record2.input = new NeuronValue[] {new NeuronValue1(99), new NeuronValue1(88), new NeuronValue1(77)};
//			record2.output = new NeuronValue[] {new NeuronValue1(66), new NeuronValue1(55), new NeuronValue1(44)};
//
//			network.learn(Arrays.asList(record1, record2));
//			
//			System.out.println(network.toString());
//		}
//		catch (Exception e) {}
		
		List<Raster> rasters = RasterAssoc.load(Paths.get("D:/1/cifar10"));
		System.out.println(rasters.size());
//		int count = RasterAssoc.saveDirector(rasters, Paths.get("D:/1/gen"), "cifar", true);
//		System.out.println(count);
	}

	
}
