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

#include <sstream>
#include <set>
#include <iomanip>
#include <algorithm>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <string.h>
#include <pthread.h>

#include "randomforest.h"


typedef std::set<ml_uint> rf_oob_indices;

#define RF_MIN_LEAF_INSTANCES 2
#define RF_BASEINFO_FILE ml_string("rf.json")
#define RF_MLID_FILE ml_string("mlid.json")


void _rf_initOutOfBagIndices(ml_uint size, rf_oob_indices &oob) {
  oob.clear();
  for(ml_uint ii=0; ii < size; ++ii) {
    oob.insert(oob.end(), ii);
  }
}

void _rf_bootstrappedSampleFromData(const ml_data &mld, ml_rng_config *rng_config, ml_data &bootstrapped, rf_oob_indices &oob) {

  bootstrapped.clear();
  _rf_initOutOfBagIndices(mld.size(), oob);

  for(std::size_t ii=0; ii < mld.size(); ++ii) {

    ml_uint index = ml_generateRandomNumber(rng_config) % mld.size();
    bootstrapped.push_back(mld[index]);

    rf_oob_indices::iterator oobit = oob.find(index);
    if(oobit != oob.end()) {
      oob.erase(oobit);
    }
  }

}

void _rf_calculateOutOfBagError(const ml_instance_definition &mlid, const ml_data &mld, const rf_forest &forest, const ml_vector<rf_oob_indices> &oobs, ml_vector<ml_feature_value> *oob_for_mld) {
  
  ml_classification_results mlcr = {};
  ml_regression_results mlrr = {};

  for(std::size_t instance_index=0; instance_index < mld.size(); ++instance_index) {
    
    rf_forest oob_forest;
    oob_forest.index_of_feature_to_predict = forest.index_of_feature_to_predict;
    oob_forest.type = forest.type;

    // find all trees that were built without this instance (it's in their out-of-bag set)
    for(std::size_t tree_index = 0; tree_index < forest.trees.size(); ++tree_index) {
      if(oobs[tree_index].find(instance_index) != oobs[tree_index].end()) {
	oob_forest.trees.push_back(forest.trees[tree_index]);
      }
    }

    ml_feature_value prediction;
    if(rf_evaluateRandomForestForInstance(mlid, oob_forest, *mld[instance_index], prediction)) {
      switch(oob_forest.type) {
      case DT_TREE_TYPE_CLASSIFICATION: ml_collectClassificationResultForInstance(mlid, oob_forest.index_of_feature_to_predict, *mld[instance_index], &prediction, mlcr); break;
      case DT_TREE_TYPE_REGRESSION: ml_collectRegressionResultForInstance(mlid, oob_forest.index_of_feature_to_predict, *mld[instance_index], &prediction, mlrr); break;
      default: break;
      }

      if(oob_for_mld) {
	oob_for_mld->push_back(prediction);
      }
    }
    
  } // for( all instances in mld )

  ml_log("\n*** Out Of Bag Error ***\n");

  switch(forest.type) {
  case DT_TREE_TYPE_CLASSIFICATION: ml_printClassificationResultsSummary(mlid, forest.index_of_feature_to_predict, mlcr); break;
  case DT_TREE_TYPE_REGRESSION: ml_printRegressionResultsSummary(mlrr); break;
  default: break;
  }
}

void _rf_collectFeatureImportance(const ml_vector<dt_feature_importance> &tree_feature_importance, ml_vector<dt_feature_importance> &forest_feature_importance) {
  for(std::size_t ii = 0; ii < tree_feature_importance.size(); ++ii) {
    forest_feature_importance[ii].count += tree_feature_importance[ii].count;
    forest_feature_importance[ii].sum_score_delta += tree_feature_importance[ii].sum_score_delta;
  }
}

void _rf_printFeatureImportance(const ml_instance_definition &mlid, ml_uint index_of_feature_to_predict, ml_vector<dt_feature_importance> &forest_feature_importance) {

  ml_double best_score_delta = 0.0;
  for(std::size_t ii = 0; ii < forest_feature_importance.size(); ++ii) {
    ml_double score_delta = (forest_feature_importance[ii].count > 0) ? forest_feature_importance[ii].sum_score_delta : 0.0;
    if(score_delta > best_score_delta) {
      best_score_delta = score_delta;
    }
  }

  ml_vector<ml_string> feature_importance_norm;
  feature_importance_norm.reserve(forest_feature_importance.size());
  for(std::size_t ii = 0; ii < forest_feature_importance.size(); ++ii) {
    if(ii == index_of_feature_to_predict) {
      continue;
    }

    ml_double score_delta = (forest_feature_importance[ii].count > 0) ? forest_feature_importance[ii].sum_score_delta : 0.0;
    ml_double avg_score_delta = (forest_feature_importance[ii].count > 0) ? (forest_feature_importance[ii].sum_score_delta / forest_feature_importance[ii].count) : 0.0;
    ml_double norm_score = (best_score_delta > 0.0) ? (100.0 * (score_delta / best_score_delta)) : 0.0;
    std::ostringstream ss;
    ss << std::setw(7) << std::fixed << std::setprecision(2) << norm_score << " " << mlid[ii].name << " (" << forest_feature_importance[ii].count << " nodes, " << avg_score_delta << ")";
    feature_importance_norm.push_back(ss.str());
  }

  ml_log("\n\n*** Feature Importance ***\n");
  std::sort(feature_importance_norm.begin(), feature_importance_norm.end());
  for(std::size_t ii=0; ii < feature_importance_norm.size(); ++ii) {
    ml_log("%s\n", feature_importance_norm[ii].c_str());
  }

}

void _rf_fillDTConfig(const rf_build_config &rfbc, const ml_instance_definition &mlid, ml_uint thread_index, dt_build_config &dtbc) {
  dtbc.index_of_feature_to_predict = rfbc.index_of_feature_to_predict;
  dtbc.max_tree_depth = rfbc.max_tree_depth;
  dtbc.rng_config = ml_createRngConfigWithSeed(rfbc.seed + thread_index);
  dtbc.min_leaf_instances = RF_MIN_LEAF_INSTANCES;
  dtbc.max_continuous_feature_splits = rfbc.max_continuous_feature_splits;
  dtbc.features_to_consider_per_node = rfbc.features_to_consider_per_node; 
}

bool _rf_SingleThreaded_buildRandomForest(const rf_build_config &rfbc, const ml_instance_definition &mlid, const ml_data &mld, rf_forest &forest, 
					  ml_vector<rf_oob_indices> &oobs, ml_vector<dt_feature_importance> &forest_feature_importance) {
  dt_build_config dtbc = {};
  _rf_fillDTConfig(rfbc, mlid, 0, dtbc);

  for(ml_uint ii=0; ii < rfbc.number_of_trees; ++ii) {
    
    ml_data bootstrapped;
    rf_oob_indices oob;
    
    _rf_bootstrappedSampleFromData(mld, dtbc.rng_config, bootstrapped, oob);

    ml_log("\nbuilding tree %d...\n", ii+1);

    dt_tree tree;
    if(!dt_buildDecisionTree(mlid, bootstrapped, dtbc, tree)) {
      ml_log_error("failed to build decision tree...");
      return(false);
    }

    oobs.push_back(oob);
    forest.trees.push_back(tree);
    _rf_collectFeatureImportance(tree.feature_importance, forest_feature_importance);

  }

  return(true);
}


#define RF_MAX_THREADS 20

typedef struct {

  pthread_t thread;
  ml_uint thread_index;
  ml_uint number_of_trees;
  dt_build_config dtbc;
  ml_vector<rf_oob_indices> oobs;
  ml_vector<dt_tree> trees;

  const ml_data *mld;
  const ml_instance_definition *mlid;

} rf_thread_config;

void *_rf_MultiThreaded_buildTree(void *config) {
  
  rf_thread_config *rftc = (rf_thread_config *) config;

  std::ostringstream ss;
  ss << "[thread " << rftc->thread_index << "]";
  ml_string thread_name = ss.str();

  for(ml_uint ii=0; ii < rftc->number_of_trees; ++ii) {
    ml_data bootstrapped;
    rf_oob_indices oob;
    
    _rf_bootstrappedSampleFromData(*(rftc->mld), rftc->dtbc.rng_config, bootstrapped, oob);
    
    ml_log("%s building tree %d...\n", thread_name.c_str(), ii+1);
    
    dt_tree tree;
    tree.name = thread_name;
    if(!dt_buildDecisionTree(*(rftc->mlid), bootstrapped, rftc->dtbc, tree)) {
      ml_log_error("failed to build decision tree %d-%d...\n", rftc->thread_index, ii+1);
      return(nullptr);
    }
    
    rftc->oobs.push_back(oob);
    rftc->trees.push_back(tree);
  }

  return(nullptr);
}

bool _rf_MultiThreaded_buildRandomForest(const rf_build_config &rfbc, const ml_instance_definition &mlid, const ml_data &mld, rf_forest &forest, 
					  ml_vector<rf_oob_indices> &oobs, ml_vector<dt_feature_importance> &forest_feature_importance) {

  if(rfbc.number_of_threads > RF_MAX_THREADS) {
    ml_log_error("requested # of threads (%u) exceeds max supported (%u)\n", rfbc.number_of_threads, RF_MAX_THREADS);
    return(false);
  }

  if(rfbc.number_of_threads > rfbc.number_of_trees) {
    ml_log_error("requested # of threads (%u) is greater than # of trees (%u)\n", rfbc.number_of_threads, rfbc.number_of_trees);
    return(false);
  }

  rf_thread_config thread_config[RF_MAX_THREADS];

  // init the dt_build_config, and other inputs for each thread
  for(ml_uint thread_index = 0; thread_index < rfbc.number_of_threads; ++thread_index) {
    thread_config[thread_index].thread_index = thread_index;
    thread_config[thread_index].mld = &mld;
    thread_config[thread_index].mlid = &mlid;
    thread_config[thread_index].number_of_trees = (rfbc.number_of_trees / rfbc.number_of_threads);
    thread_config[thread_index].number_of_trees += (thread_index == 0) ? (rfbc.number_of_trees % rfbc.number_of_threads) : 0;
    _rf_fillDTConfig(rfbc, mlid, thread_index, thread_config[thread_index].dtbc);
  }

  // spawn threads 
  for(ml_uint thread_index = 0; thread_index < rfbc.number_of_threads; ++thread_index) {
    pthread_create(&thread_config[thread_index].thread, nullptr, _rf_MultiThreaded_buildTree, &thread_config[thread_index]);
  }

  // wait for all threads to finish
  for(ml_uint thread_index = 0; thread_index < rfbc.number_of_threads; ++thread_index) {
    pthread_join(thread_config[thread_index].thread, nullptr);
  }
						
  // combine trees, out of bag maps, and feature importance
  for(ml_uint thread_index = 0; thread_index < rfbc.number_of_threads; ++thread_index) {

    if(thread_config[thread_index].trees.size() != thread_config[thread_index].number_of_trees) {
      ml_log_error("some trees failed to build in thread %d\n", thread_index);
      return(false);
    }

    for(ml_uint tree_index = 0; tree_index < thread_config[thread_index].number_of_trees; ++tree_index) {
      oobs.push_back(thread_config[thread_index].oobs[tree_index]);
      forest.trees.push_back(thread_config[thread_index].trees[tree_index]);
      _rf_collectFeatureImportance(thread_config[thread_index].trees[tree_index].feature_importance, forest_feature_importance);
    }

  }

  return(true);
}

bool rf_buildRandomForest(const ml_instance_definition &mlid, const ml_data &mld, const rf_build_config &rfbc, rf_forest &forest, ml_vector<ml_feature_value> *oob_for_mld) {

  if(mlid.empty()) {
    ml_log_error("invalid instance definition...\n");
    return(false);
  }

  forest.trees.clear();
  forest.index_of_feature_to_predict = rfbc.index_of_feature_to_predict;
  forest.type = (mlid[rfbc.index_of_feature_to_predict].type == ML_FEATURE_TYPE_DISCRETE) ? DT_TREE_TYPE_CLASSIFICATION : DT_TREE_TYPE_REGRESSION;

  ml_vector<rf_oob_indices> oobs;
  ml_vector<dt_feature_importance> forest_feature_importance;
  forest_feature_importance.resize(mlid.size());

  bool forestWasBuilt = (rfbc.number_of_threads <= 1) ? _rf_SingleThreaded_buildRandomForest(rfbc, mlid, mld, forest, oobs, forest_feature_importance) :
        _rf_MultiThreaded_buildRandomForest(rfbc, mlid, mld, forest, oobs, forest_feature_importance);

  if(!forestWasBuilt) {
    ml_log_error("hit a snag while building the forest...\n");
    return(false);
  }

  _rf_calculateOutOfBagError(mlid, mld, forest, oobs, oob_for_mld);
  _rf_printFeatureImportance(mlid, forest.index_of_feature_to_predict, forest_feature_importance);

  return(true);
}

bool _rf_evaluateClassificationRandomForestForInstance(const ml_instance_definition &mlid, const rf_forest &forest, 
						       const ml_instance &instance, ml_feature_value &prediction, ml_vector<ml_feature_value> *tree_predictions) {

  if(forest.trees.size() == 0) {
    return(false);
  }

  ml_map<ml_uint, ml_uint> prediction_map;

  for(std::size_t ii=0; ii < forest.trees.size(); ++ii) {
    const dt_tree &tree = forest.trees[ii];
    const ml_feature_value *mlfv = dt_evaluateDecisionTreeForInstance(mlid, tree, instance);
    if(mlfv) {
      prediction_map[mlfv->discrete_value_index] += 1;
      if(tree_predictions) {
	tree_predictions->push_back(*mlfv);
      }
    }
    else {
      ml_log_error("decision tree eval failed...\n");
      return(false);
    }
  }

  ml_uint predicted_discrete_value_index = 0;
  ml_uint predicted_count = 0;
  for(ml_map<ml_uint, ml_uint>::iterator it = prediction_map.begin(); it != prediction_map.end(); ++it) {
    if(it->second > predicted_count) {
      predicted_discrete_value_index = it->first;
      predicted_count = it->second;
    }
  }

  prediction.discrete_value_index = predicted_discrete_value_index;

  return(true);
}

bool _rf_evaluateRegressionRandomForestForInstance(const ml_instance_definition &mlid, const rf_forest &forest, const ml_instance &instance, 
						   ml_feature_value &prediction, ml_vector<ml_feature_value> *tree_predictions) {

  if(forest.trees.size() == 0) {
    return(false);
  }

  ml_double sum = 0;

  for(std::size_t ii=0; ii < forest.trees.size(); ++ii) {
    const dt_tree &tree = forest.trees[ii];
    const ml_feature_value *mlfv = dt_evaluateDecisionTreeForInstance(mlid, tree, instance);
    if(mlfv) {
      sum += mlfv->continuous_value;
      if(tree_predictions) {
	tree_predictions->push_back(*mlfv);
      }
    }
    else {
      ml_log_error("decision tree eval failed...\n");
      return(false);
    }
  }

  prediction.continuous_value = (ml_float) (sum / forest.trees.size());

  return(true);
}

bool rf_evaluateRandomForestForInstance(const ml_instance_definition &mlid, const rf_forest &forest, const ml_instance &instance, 
					ml_feature_value &prediction, ml_vector<ml_feature_value> *tree_predictions) {
  if(tree_predictions) {
    tree_predictions->clear();
  }

  if(forest.type == DT_TREE_TYPE_CLASSIFICATION) {
    return(_rf_evaluateClassificationRandomForestForInstance(mlid, forest, instance, prediction, tree_predictions));
  }

  return(_rf_evaluateRegressionRandomForestForInstance(mlid, forest, instance, prediction, tree_predictions));
}

void rf_printRandomForestResultsForData(const ml_instance_definition &mlid, const ml_data &mld, const rf_forest &forest) {
 
  ml_classification_results mlcr = {};
  ml_regression_results mlrr = {};

  for(std::size_t instance_index=0; instance_index < mld.size(); ++instance_index) {
    
    ml_feature_value prediction;
    if(rf_evaluateRandomForestForInstance(mlid, forest, *mld[instance_index], prediction)) {
      switch(forest.type) {
      case DT_TREE_TYPE_CLASSIFICATION: ml_collectClassificationResultForInstance(mlid, forest.index_of_feature_to_predict, *mld[instance_index], &prediction, mlcr); break;
      case DT_TREE_TYPE_REGRESSION: ml_collectRegressionResultForInstance(mlid, forest.index_of_feature_to_predict, *mld[instance_index], &prediction, mlrr); break;
      default: break;
      }
    }
    
  } 

  switch(forest.type) {
  case DT_TREE_TYPE_CLASSIFICATION: ml_printClassificationResultsSummary(mlid, forest.index_of_feature_to_predict, mlcr); break;
  case DT_TREE_TYPE_REGRESSION: ml_printRegressionResultsSummary(mlrr); break;
  default: break;
  }
}

cJSON *_rf_createJSONObjectWithBaseInfoFromRandomForest(const rf_forest &forest) {
  //
  // the trees are stored elsewhere as separate json files in the chosen rf directory
  //
  cJSON *json_forest = cJSON_CreateObject();
  cJSON_AddStringToObject(json_forest, "object", "rf_forest");
  cJSON_AddNumberToObject(json_forest, "type", forest.type);
  cJSON_AddNumberToObject(json_forest, "index_of_feature_to_predict", forest.index_of_feature_to_predict);
  return(json_forest);
}

bool _rf_fillRandomForestWithBaseInfoFromJSONObject(cJSON *json_object, rf_forest &forest) {

  if(!json_object) {
    ml_log_error("nil json object...\n");
    return(false);
  }

  cJSON *object = cJSON_GetObjectItem(json_object, "object");
  if(!object || !object->valuestring || strcmp(object->valuestring, "rf_forest")) {
    ml_log_error("json object is not a random forest...\n");
    return(false);
  }

  cJSON *type = cJSON_GetObjectItem(json_object, "type");
  if(!type || (type->type != cJSON_Number)) {
    ml_log_error("json object is missing forest type\n");
    return(false);
  }

  forest.type = (dt_tree_type) type->valueint;

  cJSON *index = cJSON_GetObjectItem(json_object, "index_of_feature_to_predict");
  if(!index || (index->type != cJSON_Number)) {
    ml_log_error("json object is missing the index of the feature to predict\n");
    return(false);
  }

  forest.index_of_feature_to_predict = index->valueint;

  return(true);
}

bool _rf_writeRandomForestBaseInfoToFile(const ml_string &path_to_file, const rf_forest &forest) {
  cJSON *json_object = _rf_createJSONObjectWithBaseInfoFromRandomForest(forest);
  if(!json_object) {
    ml_log_error("couldn't create json object from random forest\n");
    return(false);
  }

  bool status = ml_writeModelJSONToFile(path_to_file, json_object);
  cJSON_Delete(json_object);

  return(status);
}

bool _rf_readRandomForestBaseInfoFromFile(const ml_string &path_to_file, rf_forest &forest) {
  cJSON *json_object = ml_readModelJSONFromFile(path_to_file);
  if(!json_object) {
    ml_log_error("couldn't load random forest json object from model file: %s\n", path_to_file.c_str());
    return(false);
  }

  bool status = _rf_fillRandomForestWithBaseInfoFromJSONObject(json_object, forest);
  cJSON_Delete(json_object);

  return(status);
}


bool rf_writeRandomForestToDirectory(const ml_string &path_to_dir, const ml_instance_definition &mlid, const rf_forest &forest, bool overwrite_existing) {
  
  if((path_to_dir == ".") || (path_to_dir == "..")) {
    return(false);
  }
  
  // clear the directory if it already exists and we are allowed to overwrite.
  struct stat info;
  if((stat(path_to_dir.c_str(), &info) == 0) && (info.st_mode & S_IFDIR)) {

    if(!overwrite_existing) {
      ml_log_error("directory exists and we aren't allowed to overwrite\n");
      return(false);
    }

    ml_string rm_cmd = ml_string("rm -f ") + path_to_dir + "/*.json";
    if(system(rm_cmd.c_str())){
      ml_log_error("couldn't clear rf directory: %s\n", path_to_dir.c_str());
      return(false);
    }
  }
  else {
    // the directory doesn't exist so we will create it.
    if(mkdir(path_to_dir.c_str(), 0755)) {
      ml_log_error("couldn't create rf directory: %s\n", path_to_dir.c_str());
      perror("ERROR --> mkdir");
      return(false);
    }
  }

  // write the instance definition
  if(!ml_writeInstanceDefinitionToFile(path_to_dir + "/" + RF_MLID_FILE, mlid)) {
    ml_log_error("couldn't write rf instance definition to %s\n", RF_MLID_FILE.c_str());
    return(false);
  }

  // store the forest type, index to predict, etc
  if(!_rf_writeRandomForestBaseInfoToFile(path_to_dir + "/" + RF_BASEINFO_FILE, forest)) {
    ml_log_error("couldn't write rf info to %s\n", RF_BASEINFO_FILE.c_str());
    return(false);
  }

  std::time_t timestamp = std::time(0); 

  // each tree is written to its own file
  for(std::size_t ii = 0; ii < forest.trees.size(); ++ii) {
    std::ostringstream ss;
    // we use the timestamp in the filename to make it easier to consolidate trees from multiple runs.
    // for example, tree1.1457973944.json
    ss << path_to_dir << "/" << "tree" << (ii+1) << "." << timestamp << ".json";
    if(!dt_writeDecisionTreeToFile(ss.str(), forest.trees[ii])) {
      ml_log_error("couldn't write tree to file: %s\n", ss.str().c_str());
      return(false);
    }
  }
 
  return(true);
  
}

bool rf_readRandomForestFromDirectory(const ml_string &path_to_dir, ml_instance_definition &mlid, rf_forest &forest) {

  if(!ml_readInstanceDefinitionFromFile(path_to_dir + "/" + RF_MLID_FILE, mlid)) {
    ml_log_error("couldn't read rf instance defintion\n");
    return(false);
  }

  if(!_rf_readRandomForestBaseInfoFromFile(path_to_dir + "/" + RF_BASEINFO_FILE, forest)) {
    ml_log_error("couldn't read rf base info\n");
    return(false);
  }

  DIR *d = 0;
  struct dirent *dir = 0;
  d = opendir(path_to_dir.c_str());
  if(d) {
    while((dir = readdir(d)) != NULL) {
      ml_string file_name(dir->d_name);
      if((file_name == ".") || (file_name == "..") || 
	 (file_name == RF_BASEINFO_FILE) || (file_name == RF_MLID_FILE)) {
	continue;
      }

      ml_string full_path = path_to_dir + "/" + dir->d_name;
      dt_tree tree = {};
      if(!dt_readDecisionTreeFromFile(full_path, tree)) {
	ml_log_error("failed to parse tree from json: %s\n", full_path.c_str());
	return(false);
      }

      forest.trees.push_back(tree);
    }

    closedir(d);
  }

  return(true);
}

void rf_freeRandomForest(rf_forest &forest) {
  for(std::size_t ii = 0; ii < forest.trees.size(); ++ii) {
    dt_freeDecisionTree(forest.trees[ii]);
  }

  forest.trees.clear();
  forest.index_of_feature_to_predict = 0;
}
