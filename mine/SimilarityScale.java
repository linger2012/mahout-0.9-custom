package examples;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;


public class SimilarityScale {

	  public static class SimilarityMapper extends Mapper<Object, Text, Text, FloatWritable>
	  {
		    
		    private Text pair = new Text();
		    private FloatWritable similarity = new FloatWritable();
		    private float scale;  
		    
		    @Override  
		      protected void setup(Context context) 
		    {
		        Configuration conf = context.getConfiguration();  
		        scale = conf.getFloat("scale", 1);
		    }
		    
		    
		    public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
		    {
		    	 String[] fields = value.toString().split("\t");
			      if(fields.length<3) return;			      
			      String cur_pair = fields[0]+"\t"+fields[1];
			      pair.set(cur_pair);
			      
			      similarity.set(Float.parseFloat(fields[2])*scale);	    	 
			      context.write(pair, similarity);		      
		    }		    
		  }
	
	
	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException 
	{
		// TODO Auto-generated method stub
	    Configuration conf = new Configuration();
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 3) 
	    {
	      System.err.println("Usage: similarityMerge <in> <out> <scale>");
	      System.exit(2);
	    }
	    
	    conf.setFloat("scale",Float.parseFloat(otherArgs[2]));
	    
	    Job job = new Job(conf, "similarityMerge");
	    
	    job.setJarByClass(SimilarityScale.class);
	    job.setMapperClass(SimilarityMapper.class);

	    
	    job.setInputFormatClass(TextInputFormat.class);
	    
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(FloatWritable.class);
	    
	    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
	    
	    
	    
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
	
	
	
	

}
