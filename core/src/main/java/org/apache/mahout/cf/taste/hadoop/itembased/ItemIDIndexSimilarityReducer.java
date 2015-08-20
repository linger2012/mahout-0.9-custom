package org.apache.mahout.cf.taste.hadoop.itembased;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.Vectors;

public class ItemIDIndexSimilarityReducer extends
Reducer<IntWritable, VectorWritable, IntWritable,VectorWritable> {
	
	
	  @Override
	  protected void reduce(IntWritable index,
	                        Iterable<VectorWritable> partialVectors,
	                        Context context) throws IOException, InterruptedException {		
		  
		  Vector partialVector = Vectors.merge(partialVectors);	  
		  context.write(index, new VectorWritable(partialVector));
		  
	  }

}
