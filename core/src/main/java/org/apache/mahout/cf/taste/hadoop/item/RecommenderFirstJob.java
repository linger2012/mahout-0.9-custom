/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.hadoop.item;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.preparation.PreparePreferenceMatrixJob;
import org.apache.mahout.cf.taste.hadoop.similarity.item.ItemSimilarityJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * <p>Runs a completely distributed recommender job as a series of mapreduces.</p>
 * <p/>
 * <p>Preferences in the input file should look like {@code userID, itemID[, preferencevalue]}</p>
 * <p/>
 * <p>
 * Preference value is optional to accommodate applications that have no notion of a preference value (that is, the user
 * simply expresses a preference for an item, but no degree of preference).
 * </p>
 * <p/>
 * <p>
 * The preference value is assumed to be parseable as a {@code double}. The user IDs and item IDs are
 * parsed as {@code long}s.
 * </p>
 * <p/>
 * <p>Command line arguments specific to this class are:</p>
 * <p/>
 * <ol>
 * <li>--input(path): Directory containing one or more text files with the preference data</li>    输入目录
 * <li>--output(path): output path where recommender output should go</li>    输出目录 
 * <li>--similarityClassname (classname): Name of vector similarity class to instantiate or a predefined similarity
 * from {@link org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasure}</li>  相似度算法名
 * <li>--usersFile (path): only compute recommendations for user IDs contained in this file (optional)</li>        指定需要推荐的user列表
 * <li>--itemsFile (path): only include item IDs from this file in the recommendations (optional)</li>     指定需要推荐的item列表
 * <li>--filterFile (path): file containing comma-separated userID,itemID pairs. Used to exclude the item from the
 * recommendations for that user (optional)</li>                   user,item黑名单过滤
 * <li>--numRecommendations (integer): Number of recommendations to compute per user (10)</li>    每个user需要推荐item的个数
 * <li>--booleanData (boolean): Treat input data as having no pref values (false)</li>      默认视为有偏好
 * <li>--maxPrefsPerUser (integer): Maximum number of preferences considered per user in final
 *   recommendation phase (10)</li>   只取该user的N个评分作为推荐依据
 * <li>--maxSimilaritiesPerItem (integer): Maximum number of similarities considered per item (100)</li>   每个item默认只有100个相似item
 * <li>--minPrefsPerUser (integer): ignore users with less preferences than this in the similarity computation (1)</li>
 * <li>--maxPrefsPerUserInItemSimilarity (integer): max number of preferences to consider per user in
 *   the item similarity computation phase,
 * users with more preferences will be sampled down (1000)</li>
 * <li>--threshold (double): discard item pairs with a similarity value below this</li>
 * </ol>
 * <p/>
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 * <p/>
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public final class RecommenderFirstJob extends AbstractJob {

  public static final String BOOLEAN_DATA = "booleanData";
  public static final String DEFAULT_PREPARE_PATH = "preparePreferenceMatrix";

  private static final int DEFAULT_MAX_SIMILARITIES_PER_ITEM = 100;
  private static final int DEFAULT_MAX_PREFS = 500;
  private static final int DEFAULT_MIN_PREFS_PER_USER = 1;

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("numRecommendations", "n", "Number of recommendations per user",
            String.valueOf(AggregateAndRecommendReducer.DEFAULT_NUM_RECOMMENDATIONS));
    addOption("usersFile", null, "File of users to recommend for", null);
    addOption("itemsFile", null, "File of items to recommend for", null);
    addOption("filterFile", "f", "File containing comma-separated userID,itemID pairs. Used to exclude the item from "
            + "the recommendations for that user (optional)", null);
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());
    addOption("maxPrefsPerUser", "mxp",
            "Maximum number of preferences considered per user in final recommendation phase",
            String.valueOf(UserVectorSplitterMapper.DEFAULT_MAX_PREFS_PER_USER_CONSIDERED));
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this in the similarity computation "
            + "(default: " + DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(DEFAULT_MIN_PREFS_PER_USER));
    addOption("maxSimilaritiesPerItem", "m", "Maximum number of similarities considered per item ",
            String.valueOf(DEFAULT_MAX_SIMILARITIES_PER_ITEM));
    addOption("maxPrefsInItemSimilarity", "mpiis", "max number of preferences to consider per user or item in the "
            + "item similarity computation phase, users or items with more preferences will be sampled down (default: "
        + DEFAULT_MAX_PREFS + ')', String.valueOf(DEFAULT_MAX_PREFS));
    addOption("similarityClassname", "s", "Name of distributed similarity measures class to instantiate, " 
            + "alternatively use one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')', true);
    addOption("threshold", "tr", "discard item pairs with a similarity value below this", false);
  //  addOption("outputPathForSimilarityMatrix", "opfsm", "write the item similarity matrix to this path (optional)", false);
    addOption("randomSeed", null, "use this seed for sampling", false);
    addFlag("sequencefileOutput", null, "write the output into a SequenceFile instead of a text file");

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

  //  Path outputPath = getOutputPath();
  //  int numRecommendations = Integer.parseInt(getOption("numRecommendations"));
   // String usersFile = getOption("usersFile");
   // String itemsFile = getOption("itemsFile");
   // String filterFile = getOption("filterFile");
    boolean booleanData = Boolean.valueOf(getOption("booleanData"));
 //   int maxPrefsPerUser = Integer.parseInt(getOption("maxPrefsPerUser"));
    int minPrefsPerUser = Integer.parseInt(getOption("minPrefsPerUser"));
    int maxPrefsInItemSimilarity = Integer.parseInt(getOption("maxPrefsInItemSimilarity"));
    int maxSimilaritiesPerItem = Integer.parseInt(getOption("maxSimilaritiesPerItem"));
    String similarityClassname = getOption("similarityClassname");
    double threshold = hasOption("threshold")
        ? Double.parseDouble(getOption("threshold")) : RowSimilarityJob.NO_THRESHOLD;
    long randomSeed = hasOption("randomSeed")
        ? Long.parseLong(getOption("randomSeed")) : RowSimilarityJob.NO_FIXED_RANDOM_SEED;


    Path prepPath = getTempPath(DEFAULT_PREPARE_PATH);
    Path similarityMatrixPath = getTempPath("similarityMatrix");
   // Path explicitFilterPath = getTempPath("explicitFilterPath");
   // Path partialMultiplyPath = getTempPath("partialMultiply");

    AtomicInteger currentPhase = new AtomicInteger();

    int numberOfUsers = -1;

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {//提交 PreparePreferenceMatrixJob 任务,生成用户特征和物品特征
      ToolRunner.run(getConf(), new PreparePreferenceMatrixJob(), new String[]{
        "--input", getInputPath().toString(),
        "--output", prepPath.toString(),
        "--minPrefsPerUser", String.valueOf(minPrefsPerUser),
        "--booleanData", String.valueOf(booleanData),
        "--tempDir", getTempPath().toString(),
      });

      numberOfUsers = HadoopUtil.readInt(new Path(prepPath, PreparePreferenceMatrixJob.NUM_USERS), getConf());
    }


    if (shouldRunNextPhase(parsedArgs, currentPhase)) {

      /* special behavior if phase 1 is skipped */
      if (numberOfUsers == -1) {
        numberOfUsers = (int) HadoopUtil.countRecords(new Path(prepPath, PreparePreferenceMatrixJob.USER_VECTORS),
                PathType.LIST, null, getConf());
      }

      //calculate the co-occurrence matrix  计算物品相似度
      ToolRunner.run(getConf(), new RowSimilarityJob(), new String[]{
        "--input", new Path(prepPath, PreparePreferenceMatrixJob.RATING_MATRIX).toString(),
        "--output", similarityMatrixPath.toString(),
        "--numberOfColumns", String.valueOf(numberOfUsers),
        "--similarityClassname", similarityClassname,
        "--maxObservationsPerRow", String.valueOf(maxPrefsInItemSimilarity),
        "--maxObservationsPerColumn", String.valueOf(maxPrefsInItemSimilarity),
        "--maxSimilaritiesPerRow", String.valueOf(maxSimilaritiesPerItem),
        "--excludeSelfSimilarity", String.valueOf(Boolean.TRUE),
        "--threshold", String.valueOf(threshold),
        "--randomSeed", String.valueOf(randomSeed),
        "--tempDir", getTempPath().toString(),
      });

      //保存item相似度矩阵,把item的ID从内部索引映射回原始ID
      // write out the similarity matrix if the user specified that behavior
     // if (hasOption("outputPathForSimilarityMatrix")) {
        Path outputPathForSimilarityMatrix = getOutputPath();//new Path(getOption("outputPathForSimilarityMatrix"));

        //ItemSimilarityJob.MostSimilarItemPairsMapper 做了截断,取了topK
        //ItemSimilarityJob.SimilarItemPairsMapper  我重写的mapper,保留所有非0相似度   (发现其实前一步的RowSimilarityJob也作了topK操作)
        //至于两种做法对最终的推荐效果的影响的好坏,有待实践证明
        //从效率上讲,取topK肯定是有优势的
        Job outputSimilarityMatrix = prepareJob(similarityMatrixPath, outputPathForSimilarityMatrix,
            SequenceFileInputFormat.class, ItemSimilarityJob.SimilarItemPairsMapper.class,
            EntityEntityWritable.class, DoubleWritable.class, ItemSimilarityJob.MostSimilarItemPairsReducer.class,
            EntityEntityWritable.class, DoubleWritable.class, TextOutputFormat.class);

        Configuration mostSimilarItemsConf = outputSimilarityMatrix.getConfiguration();
        mostSimilarItemsConf.set(ItemSimilarityJob.ITEM_ID_INDEX_PATH_STR,
            new Path(prepPath, PreparePreferenceMatrixJob.ITEMID_INDEX).toString());
        mostSimilarItemsConf.setInt(ItemSimilarityJob.MAX_SIMILARITIES_PER_ITEM, maxSimilaritiesPerItem);
        outputSimilarityMatrix.waitForCompletion(true);
      //}
    }



    return 0;
  }
  



  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new RecommenderFirstJob(), args);
  }
}
