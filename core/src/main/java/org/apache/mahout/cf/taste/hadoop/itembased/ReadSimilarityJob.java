package org.apache.mahout.cf.taste.hadoop.itembased;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;

public class ReadSimilarityJob extends AbstractJob {

	@Override
	public int run(String[] args) throws Exception {
		
	    addInputOption();
	    addOutputOption();
	    
	    Map<String, List<String>> parsedArgs = parseArguments(args);
	    if (parsedArgs == null) {
	      return -1;
	    }
	    
	    
	    Job  itemIDIndexSimilarityJob = prepareJob(getInputPath(),getOutputPath(),
	    TextInputFormat.class,ItemIDIndexSimilarityMapper.class,IntWritable.class,VectorWritable.class,
	    ItemIDIndexSimilarityReducer.class,IntWritable.class,VectorWritable.class,SequenceFileOutputFormat.class);
	    itemIDIndexSimilarityJob.setCombinerClass(ItemIDIndexSimilarityReducer.class);
	    
	    
	    boolean succeeded = itemIDIndexSimilarityJob.waitForCompletion(true);
	    if (!succeeded)
	    {
	      return -1;
	    }
			
		return 0;
	}

}
