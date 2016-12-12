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

#include "boosting.h"
#include "mlutil.h"

#include <sstream>
#include <string.h>

namespace puml {

const ml_string &BOOSTED_BASEINFO_FILE = "boosted.json";
const ml_string &BOOSTED_MLID_FILE = "mlid.json";


static void fillDTConfig(dt_build_config &boosted_dtbc, const boosted_build_config &bbc) {

  const ml_uint DEFAULT_BOOSTING_MIN_LEAF_INSTANCES = 2;
  const ml_uint DEFAULT_BOOSTING_MAX_CONT_FEATURE_SPLITS = 40;

  boosted_dtbc.max_tree_depth = bbc.max_tree_depth;;
  boosted_dtbc.min_leaf_instances = (bbc.min_leaf_instances == 0) ? DEFAULT_BOOSTING_MIN_LEAF_INSTANCES : bbc.min_leaf_instances;
  boosted_dtbc.max_continuous_feature_splits = DEFAULT_BOOSTING_MAX_CONT_FEATURE_SPLITS;
  boosted_dtbc.rng_config = createRngConfigWithSeed(bbc.seed);
  boosted_dtbc.index_of_feature_to_predict = bbc.index_of_feature_to_predict;
  boosted_dtbc.features_to_consider_per_node = bbc.features_to_consider_per_node;

}


static void sampleWithoutReplacement(const ml_mutable_data &mld, ml_data &mld_iter, ml_float subsample, ml_rng_config *rng_config) {

  mld_iter.clear();

  if(!rng_config) {
    return;
  }

  ml_uint thresh = subsample * 10000.0;
  thresh = (thresh == 0) ? 5000 : thresh;

  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    if((generateRandomNumber(rng_config) % 10000) < thresh) {
      mld_iter.push_back(mld[ii]);
    }
  }
}


bool buildBoostedTrees(const ml_instance_definition &mlid, const boosted_build_config &bbc,
		       const ml_mutable_data &mld, boosted_trees &bt, 
		       boostedBuildCallback callback, void *user) {

  bt.trees.clear();
  bt.learning_rate = bbc.learning_rate;
  bt.index_of_feature_to_predict = bbc.index_of_feature_to_predict;
  bt.type = (mlid[bbc.index_of_feature_to_predict].type == ML_FEATURE_TYPE_DISCRETE) ? DT_TREE_TYPE_CLASSIFICATION : DT_TREE_TYPE_REGRESSION;
  
  if(bt.type != DT_TREE_TYPE_REGRESSION) {
    log_error("boosting only implemented for regression...\n");
    return(false);
  }

  dt_build_config boosted_dtbc = {};
  fillDTConfig(boosted_dtbc, bbc);

  ml_vector<ml_feature_value> original_target;
  original_target.reserve(mld.size());

  for(ml_uint ii=0; ii < bbc.number_of_trees; ++ii) {

    dt_tree boosted_tree = {};

    ml_data mld_iter;    
    sampleWithoutReplacement(mld, mld_iter, bbc.subsample, boosted_dtbc.rng_config);

    log("\nbuilding boosted tree %u\n", ii+1);
    if(!buildDecisionTree(mlid, mld_iter, boosted_dtbc, boosted_tree)) {
      log_error("failed to build boosted tree...\n");
      return(false);
    }

    //printDecisionTreeSummary(mlid, boosted_tree);
    bt.trees.push_back(boosted_tree);

    //
    // update the feature to predict with the residual
    //
    for(std::size_t jj=0; jj < mld.size(); ++jj) {
      ml_instance *instance = mld[jj];
      ml_feature_value &fv = (*instance)[boosted_tree.index_of_feature_to_predict];
    
      //
      // store the original target value on the first iteration
      //
      if(ii == 0) {
	original_target.push_back(fv);
      }

      const ml_feature_value *predicted = evaluateDecisionTreeForInstance(mlid, boosted_tree, *instance);
      ml_double residual = ml_double(fv.continuous_value) - (bbc.learning_rate * predicted->continuous_value);
      fv.continuous_value = residual;
    }

    //
    // if a progress callback was given, exercise
    // and make this the final boosting iteration if 
    // it returns false
    //
    if(callback) {
      if(!callback(mlid, bt, ii+1, user)) {
	break;
      }
    }

  }


  //
  // restore the original target value
  //
  for(std::size_t instance_index=0; instance_index < mld.size(); ++instance_index) {
    (*mld[instance_index])[bbc.index_of_feature_to_predict] = original_target[instance_index];
  }

  delete boosted_dtbc.rng_config;

  return(true);
}


void freeBoostedTrees(boosted_trees &bt) {
  for(std::size_t ii = 0; ii < bt.trees.size(); ++ii) {
    freeDecisionTree(bt.trees[ii]);
  }

  bt.trees.clear();
}


bool evaluateBoostedTreesForInstance(const ml_instance_definition &mlid, const boosted_trees &bt, 
				     const ml_instance &instance, ml_feature_value &prediction) {

  if(bt.trees.size() == 0) {
    return(false);
  }

  ml_double ensemble_prediction = 0;
  for(std::size_t bb = 0; bb < bt.trees.size(); ++bb) {
    const ml_feature_value *tree_prediction = evaluateDecisionTreeForInstance(mlid, bt.trees[bb], instance);
    ensemble_prediction += (bt.learning_rate * tree_prediction->continuous_value);
  }

  prediction.continuous_value = ensemble_prediction;

  return(true);
}


void printBoostedTreesResultsForData(const ml_instance_definition &mlid, 
				     const ml_data &mld, const boosted_trees &bt) {

  ml_regression_results mlrr = {};

  for(std::size_t instance_index=0; instance_index < mld.size(); ++instance_index) {
    ml_feature_value prediction;
    if(evaluateBoostedTreesForInstance(mlid, bt, *mld[instance_index], prediction)) {
      collectRegressionResultForInstance(mlid, bt.index_of_feature_to_predict, *mld[instance_index], &prediction, mlrr); 
    }
  }

  printRegressionResultsSummary(mlrr);

}

static cJSON *createJSONObjectWithBaseInfoFromBoostedTrees(const boosted_trees &bt) {
  cJSON *json_boosted = cJSON_CreateObject();
  cJSON_AddStringToObject(json_boosted, "object", "boosted_trees");
  cJSON_AddNumberToObject(json_boosted, "type", bt.type);
  cJSON_AddNumberToObject(json_boosted, "index_of_feature_to_predict", bt.index_of_feature_to_predict);
  cJSON_AddNumberToObject(json_boosted, "learning_rate", bt.learning_rate);
  return(json_boosted);
}

static bool writeBoostedTreesBaseInfoToFile(const ml_string &path_to_file, const boosted_trees &bt) {
  cJSON *json_object = createJSONObjectWithBaseInfoFromBoostedTrees(bt);
  if(!json_object) {
    log_error("couldn't create json object from boosted trees\n");
    return(false);
  }

  bool status = writeModelJSONToFile(path_to_file, json_object);
  cJSON_Delete(json_object);

  return(status);
}

bool writeBoostedTreesToDirectory(const ml_string &path_to_dir, const ml_instance_definition &mlid, 
				  const boosted_trees &bt, bool overwrite_existing) {

  prepareDirectoryForModelSave(path_to_dir, overwrite_existing);

  // write the instance definition
  if(!writeInstanceDefinitionToFile(path_to_dir + "/" + BOOSTED_MLID_FILE, mlid)) {
    log_error("couldn't write boosted instance definition to %s\n", BOOSTED_MLID_FILE.c_str());
    return(false);
  }

  // store the tree type, index to predict, and learning rate
  if(!writeBoostedTreesBaseInfoToFile(path_to_dir + "/" + BOOSTED_BASEINFO_FILE, bt)) {
    log_error("couldn't write boosted info to %s\n", BOOSTED_BASEINFO_FILE.c_str());
    return(false);
  }

  // each tree in the ensemble is written to its own file
  for(std::size_t ii = 0; ii < bt.trees.size(); ++ii) {
    std::ostringstream ss;
    ss << path_to_dir << "/" << puml::TREE_MODEL_FILE_PREFIX << (ii+1) << ".json";
    if(!writeDecisionTreeToFile(ss.str(), bt.trees[ii])) {
      log_error("couldn't write boosted tree to file: %s\n", ss.str().c_str());
      return(false);
    }
  }

  return(true);
}


static bool fillBoostedTreesWithBaseInfoFromJSONObject(cJSON *json_object, boosted_trees &bt) {

  if(!json_object) {
    log_error("nil json object...\n");
    return(false);
  }

  cJSON *object = cJSON_GetObjectItem(json_object, "object");
  if(!object || !object->valuestring || strcmp(object->valuestring, "boosted_trees")) {
    log_error("json object is not a boosted tree ensemble...\n");
    return(false);
  }

  cJSON *type = cJSON_GetObjectItem(json_object, "type");
  if(!type || (type->type != cJSON_Number)) {
    log_error("json object is missing forest type\n");
    return(false);
  }

  bt.type = (dt_tree_type) type->valueint;

  cJSON *index = cJSON_GetObjectItem(json_object, "index_of_feature_to_predict");
  if(!index || (index->type != cJSON_Number)) {
    log_error("json object is missing the index of the feature to predict\n");
    return(false);
  }

  bt.index_of_feature_to_predict = index->valueint;

  cJSON *learning_rate = cJSON_GetObjectItem(json_object, "learning_rate");
  if(!index || (index->type != cJSON_Number)) {
    log_error("json object is missing the learning rate\n");
    return(false);
  }

  bt.learning_rate = learning_rate->valuedouble;

  return(true);
}

static bool readBoostedTreesBaseInfoFromFile(const ml_string &path_to_file, boosted_trees &bt) {
  cJSON *json_object = readModelJSONFromFile(path_to_file);
  if(!json_object) {
    log_error("couldn't load boosted trees json object from model file: %s\n", path_to_file.c_str());
    return(false);
  }

  bool status = fillBoostedTreesWithBaseInfoFromJSONObject(json_object, bt);
  cJSON_Delete(json_object);

  return(status);
}


bool readBoostedTreesFromDirectory(const ml_string &path_to_dir, ml_instance_definition &mlid, boosted_trees &bt) {

  if(!readInstanceDefinitionFromFile(path_to_dir + "/" + BOOSTED_MLID_FILE, mlid)) {
    log_error("couldn't read boosted trees instance defintion\n");
    return(false);
  }

  if(!readBoostedTreesBaseInfoFromFile(path_to_dir + "/" + BOOSTED_BASEINFO_FILE, bt)) {
    log_error("couldn't read boosted trees base info\n");
    return(false);
  }

  readDecisionTreesFromDirectory(path_to_dir, bt.trees);

  return(true);
}



} // namespace puml
