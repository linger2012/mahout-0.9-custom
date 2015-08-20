package org.apache.mahout.cf.taste.hadoop.itembased;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;

public class ItemIDIndexSimilarityMapper extends
Mapper<LongWritable,Text, IntWritable, VectorWritable>{

	  @Override
	  protected void map(LongWritable key,
	                     Text value,
	                     Context context) throws IOException, InterruptedException {
		  
		  String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());
		  long item1 =Long.parseLong(tokens[0]);
		  long item2 =Long.parseLong(tokens[1]);
		  float similarity = Float.parseFloat(tokens[2]);
		  
		  int index1 = TasteHadoopUtils.idToIndex(item1);//暂时先这样,先别过多考虑冲突的问题.看看效果再说
		  int index2 = TasteHadoopUtils.idToIndex(item2);
		  
	      RandomAccessSparseVector partialVector = new RandomAccessSparseVector(Integer.MAX_VALUE);
	      partialVector.setQuick(index1, similarity);
	      
	      context.write(new IntWritable(index2), new VectorWritable(partialVector));
	      
	      partialVector = new RandomAccessSparseVector(Integer.MAX_VALUE);
	      partialVector.setQuick(index2, similarity);
		  
	      context.write(new IntWritable(index1), new VectorWritable(partialVector));
	      
	      
	      
	  }
	
}
