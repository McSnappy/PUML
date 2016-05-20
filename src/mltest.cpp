/*
Copyright (c) 2016 Carl Sherrell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdlib.h>
#include <unistd.h>

#include "randomforest.h"

int main(int argc, char **argv) {

  //
  // Decision Tree Example
  //

  // Load the Iris dataset 
  ml_data iris_mld;
  ml_instance_definition iris_mlid;
  ml_loadInstanceDataFromFile("./iris.csv", iris_mlid, iris_mld);

  // Take 50% for training
  ml_data iris_training, iris_test;
  ml_rng_config *rng_config = ml_createRngConfigWithSeed(333);
  ml_splitDataIntoTrainingAndTest(iris_mld, 0.5, rng_config, iris_training, iris_test);

  // Build the tree
  dt_tree iris_tree;
  dt_build_config dtbc = {};
  dtbc.max_tree_depth = 6;
  dtbc.min_leaf_instances = 2;
  dtbc.index_of_feature_to_predict = ml_indexOfFeatureWithName("Class", iris_mlid);
  if(!dt_buildDecisionTree(iris_mlid, iris_training, dtbc, iris_tree)) {
    ml_log_error("failed to build tree...\n");
    exit(1);
  }

  // Show the tree
  dt_printDecisionTreeSummary(iris_mlid, iris_tree);

  // Show the results over the hold out data
  dt_printDecisionTreeResultsForData(iris_mlid, iris_test, iris_tree);

  ml_freeInstanceData(iris_training);
  ml_freeInstanceData(iris_test);
  dt_freeDecisionTree(iris_tree);


  //
  // Random Forest Example 
  //
  ml_data cover_mld;
  ml_instance_definition cover_mlid;
  ml_loadInstanceDataFromFile("./covertype.csv", cover_mlid, cover_mld);
  ml_printInstanceDataSummary(cover_mlid);

  // Take 10% for training just for demonstration purposes
  ml_data cover_training, cover_test;
  ml_splitDataIntoTrainingAndTest(cover_mld, 0.1, rng_config, cover_training, cover_test);

  rf_forest cover_forest;
  rf_build_config rfbc = {};
  rfbc.index_of_feature_to_predict = ml_indexOfFeatureWithName("CoverType", cover_mlid);
  rfbc.number_of_trees = 50;
  rfbc.number_of_threads = 2;
  rfbc.max_tree_depth = 30;
  rfbc.seed = 999;
  rfbc.max_continuous_feature_splits = 20; // experimental optimization
  rfbc.features_to_consider_per_node = (ml_uint)(sqrt(cover_mlid.size()-1) + 0.5);
 
  if(!rf_buildRandomForest(cover_mlid, cover_training, rfbc, cover_forest)) {
    ml_log_error("failed to build random forest...\n");
    exit(1);
  }
  
  // Show results of forest for the hold out
  rf_printRandomForestResultsForData(cover_mlid, cover_test, cover_forest);

  // Save the model to disk
  rf_writeRandomForestToDirectory("./rf-cover", cover_mlid, cover_forest);

  ml_freeInstanceData(cover_training);
  ml_freeInstanceData(cover_test);
  rf_freeRandomForest(cover_forest);

  return(0);
}
