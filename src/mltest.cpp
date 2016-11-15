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
#include "boosting.h"


int main(int argc, char **argv) {

  //
  // Decision Tree Example
  //
  puml::log("\n\n *** Single Decision Tree Example Using Iris Data From UCI ***\n");

  // Load the Iris dataset 
  puml::ml_mutable_data iris_mld;
  puml::ml_instance_definition iris_mlid;
  puml::loadInstanceDataFromFile("./iris.csv", iris_mlid, iris_mld);

  // Take 50% for training
  puml::ml_mutable_data iris_training, iris_test;
  puml::ml_rng_config *rng_config = puml::createRngConfigWithSeed(333);
  puml::splitDataIntoTrainingAndTest(iris_mld, 0.5, rng_config, iris_training, iris_test);

  // Build the tree
  puml::dt_tree iris_tree;
  puml::dt_build_config dtbc = {};
  dtbc.max_tree_depth = 6;
  dtbc.min_leaf_instances = 2;
  dtbc.index_of_feature_to_predict = indexOfFeatureWithName("Class", iris_mlid);
  if(!puml::buildDecisionTree(iris_mlid, puml::ml_data(iris_training.begin(), iris_training.end()), dtbc, iris_tree)) {
    puml::log_error("failed to build tree...\n");
    exit(1);
  }

  // Show the tree
  puml::printDecisionTreeSummary(iris_mlid, iris_tree);

  // Show the results over the hold out data
  puml::printDecisionTreeResultsForData(iris_mlid, puml::ml_data(iris_test.begin(), iris_test.end()), iris_tree);

  // Release resources
  puml::freeInstanceData(iris_training);
  puml::freeInstanceData(iris_test);
  puml::freeDecisionTree(iris_tree);


  //
  // Random Forest Example 
  //
  puml::log("\n\n *** Random Forest Example Using CoverType Data From UCI ***\n");
  puml::ml_mutable_data cover_mld;
  puml::ml_instance_definition cover_mlid;
  puml::loadInstanceDataFromFile("./covertype.csv", cover_mlid, cover_mld);
  puml::printInstanceDataSummary(cover_mlid);

  // Train on 10% for this demonstration 
  puml::ml_mutable_data cover_training, cover_test;
  puml::splitDataIntoTrainingAndTest(cover_mld, 0.1, rng_config, cover_training, cover_test);
  
  puml::rf_forest cover_forest;
  puml::rf_build_config rfbc = {};
  rfbc.index_of_feature_to_predict = puml::indexOfFeatureWithName("CoverType", cover_mlid);
  rfbc.number_of_trees = 50;
  rfbc.number_of_threads = 2;
  rfbc.max_tree_depth = 30;
  rfbc.seed = 999;
  rfbc.max_continuous_feature_splits = 20; // experimental optimization
  rfbc.features_to_consider_per_node = (puml::ml_uint)(sqrt(cover_mlid.size()-1) + 0.5);
   
  if(!puml::buildRandomForest(cover_mlid, puml::ml_data(cover_training.begin(), cover_training.end()), rfbc, cover_forest)) {
    puml::log_error("failed to build random forest...\n");
    exit(1);
  }

  // Show results of forest for the hold out
  puml::printRandomForestResultsForData(cover_mlid, puml::ml_data(cover_test.begin(), cover_test.end()), cover_forest);

  // Save the model to disk
  puml::writeRandomForestToDirectory("/tmp/rf-cover", cover_mlid, cover_forest);

  // Release resources
  puml::freeInstanceData(cover_training);
  puml::freeInstanceData(cover_test);
  puml::freeRandomForest(cover_forest);
  

  //
  // Boosted Regression Trees example
  //
  puml::log("\n\n *** Boosted Trees Example Using Wine Quality Data From UCI ***\n");
  puml::ml_mutable_data wine_mld;
  puml::ml_instance_definition wine_mlid;
  puml::loadInstanceDataFromFile("./winequality-white.csv", wine_mlid, wine_mld);

  // Take 90% for training
  puml::ml_mutable_data wine_training, wine_test;
  puml::splitDataIntoTrainingAndTest(wine_mld, 0.9, rng_config, wine_training, wine_test);

  // Build the boosted trees
  puml::boosted_trees bt;
  puml::boosted_build_config bbc = {};
  bbc.learning_rate = 0.1;
  bbc.number_of_trees = 100;
  bbc.max_tree_depth = 6;
  bbc.index_of_feature_to_predict = puml::indexOfFeatureWithName("quality", wine_mlid);
  puml::buildBoostedTrees(wine_mlid, bbc, wine_training, bt);

  // Show the results using the holdout data
  puml::printBoostedTreesResultsForData(wine_mlid, puml::ml_data(wine_test.begin(), wine_test.end()), bt);

  // Store the model 
  puml::writeBoostedTreesToDirectory("/tmp/boosted_test", wine_mlid, bt);

  // Release resources
  puml::freeInstanceData(wine_training);
  puml::freeInstanceData(wine_test);
  puml::freeBoostedTrees(bt);
  

  return(0);
}
