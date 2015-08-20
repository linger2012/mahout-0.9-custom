package examples;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;



public class SimilarityMerge {

	  public static class SimilarityMapper extends Mapper<Object, Text, Text, FloatWritable>
	  {
		    
		    private Text pair = new Text();
		    private FloatWritable similarity = new FloatWritable();
		      
		    public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
		    {
		    	 String[] fields = value.toString().split("\t");
			      if(fields.length<3) return;			      
			      String cur_pair = fields[0]+"\t"+fields[1];
			      pair.set(cur_pair);
			      
			      similarity.set(Float.parseFloat(fields[2]));	    	 
			      context.write(pair, similarity);		      
		    }		    
		  }
	  
	  public static class SimilarityReducer extends Reducer<Text,FloatWritable,Text,FloatWritable> 
	  {
		private FloatWritable result = new FloatWritable();
	    public void reduce(Text key, Iterable<FloatWritable> values, Context context) throws IOException, InterruptedException 
	    {
	     float sum=0;
	      for (FloatWritable val : values) 
	      {
	    	  sum+= val.get();
	      }      
	      result.set(sum);
	      context.write(key, result);
	    }
	  }
	  
	  
	  public static void main(String[] args) throws Exception 
	  {
	    Configuration conf = new Configuration();
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 3) 
	    {
	      System.err.println("Usage: similarityMerge <in> <in> <out>");
	      System.exit(2);
	    }
	    
	    Job job = new Job(conf, "similarityMerge");
	    
	    job.setJarByClass(SimilarityMerge.class);
	    job.setMapperClass(SimilarityMapper.class);
	   job.setCombinerClass(SimilarityReducer.class);
	    job.setReducerClass(SimilarityReducer.class);
	    
	    job.setInputFormatClass(TextInputFormat.class);
	    
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(FloatWritable.class);
	    
	    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[1]));
	    FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));
	    
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	  }
	  
	  
}
