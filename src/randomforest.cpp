/*
Copyright (c) Carl Sherrell

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
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <string.h>

#include "randomforest.h"
#include "mlutil.h"

namespace puml {

static const ml_string &RF_BASEINFO_FILE = "rf.json";
static const ml_string &RF_MLID_FILE = "mlid.json";

struct rf_thread_config {

  rf_thread_config(ml_uint tindex, ml_uint ntrees,
		   const decision_tree &ptree, const ml_data &data) :
    thread_index(tindex), number_of_trees(ntrees),
    proto_tree(ptree), mld(data) {}

  ml_uint thread_index;
  ml_uint number_of_trees;
  decision_tree proto_tree;
  const ml_data &mld;

  ml_vector<rf_oob_indices> oobs;
  ml_vector<decision_tree> trees;
};

using rf_thread_config_ptr = std::shared_ptr<rf_thread_config>;


random_forest::random_forest(const ml_instance_definition &mlid,
			     const ml_string &feature_to_predict,
			     ml_uint number_of_trees,
			     ml_uint seed,
			     ml_uint number_of_threads,
			     ml_uint max_tree_depth,
			     ml_uint min_leaf_instances,
			     ml_uint features_to_consider_per_node) :
  mlid_(mlid),
  index_of_feature_to_predict_(index_of_feature_with_name(feature_to_predict, mlid)),
  number_of_trees_(number_of_trees),
  seed_(seed),
  number_of_threads_(number_of_threads),
  max_tree_depth_(max_tree_depth),
  min_leaf_instances_(min_leaf_instances),
  features_to_consider_per_node_(features_to_consider_per_node) {

  type_ = (mlid_[index_of_feature_to_predict_]->type == ml_feature_type::discrete) ? ml_model_type::classification : ml_model_type::regression;
  
  if(number_of_threads_ > number_of_trees_) {
    number_of_threads_ = 1;
  }

  if(features_to_consider_per_node_ == RF_DEFAULT_FEATURES_SQRT) {
    features_to_consider_per_node_ = (ml_uint) (std::sqrt(mlid.size() - 1) + 0.5);
  }
}


static void init_outofbag_indices(ml_uint size, rf_oob_indices &oob) {
  oob.clear();
  for(ml_uint ii=0; ii < size; ++ii) {
    oob.insert(oob.end(), ii);
  }
}


static void bootstrapped_sample_from_data(const ml_data &mld, ml_rng &rng, 
					  ml_data &bootstrapped, rf_oob_indices &oob) {

  bootstrapped.clear();
  init_outofbag_indices(mld.size(), oob);

  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    ml_uint index = rng.random_number() % mld.size();
    bootstrapped.push_back(mld[index]);

    rf_oob_indices::iterator oobit = oob.find(index);
    if(oobit != oob.end()) {
      oob.erase(oobit);
    }
  }

}


void random_forest::set_trees(const ml_vector<decision_tree> &trees) {
  oob_predictions_.clear();
  feature_importance_.clear();
  trees_ = trees;
}


void random_forest::evaluate_out_of_bag(const ml_data &mld, const ml_vector<rf_oob_indices> &oobs) {
  
  oob_predictions_.clear();
  random_forest oob_forest = *this;

  for(ml_uint instance_index = 0; instance_index < mld.size(); ++instance_index) {
    
    // find all trees that were built without this instance (it's in their out-of-bag set)
    ml_vector<decision_tree> oob_trees;
    for(std::size_t tree_index = 0; tree_index < trees_.size(); ++tree_index) {
      if(oobs[tree_index].find(instance_index) != oobs[tree_index].end()) {
	oob_trees.push_back(trees_[tree_index]);
      }
    }

    oob_forest.set_trees(oob_trees);    
    ml_feature_value prediction = oob_forest.evaluate(*mld[instance_index]);
    oob_predictions_.push_back(prediction);
  } 

}


static void collect_feature_importance(const ml_vector<dt_feature_importance> &tree_feature_importance, 
				       ml_vector<dt_feature_importance> &forest_feature_importance) {
  for(std::size_t ii = 0; ii < tree_feature_importance.size(); ++ii) {
    forest_feature_importance[ii].count += tree_feature_importance[ii].count;
    forest_feature_importance[ii].sum_score_delta += tree_feature_importance[ii].sum_score_delta;
  }
}


static ml_vector<feature_importance_tuple> calculate_feature_importance(const ml_instance_definition &mlid,
									ml_uint index_of_feature_to_predict,
									ml_vector<dt_feature_importance> forest_feature_importance) {
  ml_double best_score_delta = 0.0;
  for(const auto &feature_importance : forest_feature_importance) {
    ml_double score_delta = (feature_importance.count > 0) ? feature_importance.sum_score_delta : 0.0;
    if(score_delta > best_score_delta) {
      best_score_delta = score_delta;
    }
  }

  ml_vector<feature_importance_tuple> feature_importance_norm;
  feature_importance_norm.reserve(forest_feature_importance.size());
  for(std::size_t ii = 0; ii < forest_feature_importance.size(); ++ii) {
    if(ii == index_of_feature_to_predict) {
      continue;
    }

    ml_double score_delta = (forest_feature_importance[ii].count > 0) ? forest_feature_importance[ii].sum_score_delta : 0.0;
    ml_double avg_score_delta = (forest_feature_importance[ii].count > 0) ? (forest_feature_importance[ii].sum_score_delta / forest_feature_importance[ii].count) : 0.0;
    ml_double norm_score = (best_score_delta > 0.0) ? (100.0 * (score_delta / best_score_delta)) : 0.0;
    std::ostringstream ss;
    ss << std::setw(7) << std::fixed << std::setprecision(2) << norm_score << " " << mlid[ii]->name << " (" << forest_feature_importance[ii].count << " nodes, " << avg_score_delta << ")";
    //
    // feature_importance_tuple represents the feature index and feature 
    // importance score description
    //
    feature_importance_norm.push_back(std::make_pair(ii, ss.str()));
  }

  std::sort(feature_importance_norm.begin(), 
	    feature_importance_norm.end(), 
	    [](feature_importance_tuple const &t1, feature_importance_tuple const &t2) {
	      return(std::get<1>(t1) < std::get<1>(t2));
	    });

  return(feature_importance_norm);
}

bool random_forest::single_threaded_train(const ml_data &mld, 
					  ml_vector<rf_oob_indices> &oobs, 
					  ml_vector<dt_feature_importance> &forest_feature_importance) {

  ml_rng rng(seed_);

  for(ml_uint ii=0; ii < number_of_trees_; ++ii) {
    
    ml_data bootstrapped;
    rf_oob_indices oob;
    bootstrapped_sample_from_data(mld, rng, bootstrapped, oob);

    log("\nbuilding tree %d...\n", ii+1);

    decision_tree tree{mlid_, index_of_feature_to_predict_, 
	max_tree_depth_, min_leaf_instances_, 
        features_to_consider_per_node_, seed_};

    if(!tree.train(bootstrapped)) {
      log_error("rf failed to build decision tree...");
      return(false);
    }

    oobs.push_back(oob);
    collect_feature_importance(tree.feature_importance(), forest_feature_importance);
    trees_.push_back(tree);

  }

  return(true);
}


static void multi_threaded_work(rf_thread_config_ptr rftc) {

  ml_rng rng(rftc->proto_tree.seed());

  for(ml_uint ii=0; ii < rftc->number_of_trees; ++ii) {

    ml_data bootstrapped;
    rf_oob_indices oob; 
    bootstrapped_sample_from_data(rftc->mld, rng, bootstrapped, oob);
    
    log("%s building tree %d...\n", rftc->proto_tree.name().c_str(), ii+1);
    
    if(!rftc->proto_tree.train(bootstrapped)) {
      log_error("rf failed to build decision tree %d-%d...\n", rftc->thread_index, ii+1);
      return;
    }
    
    rftc->oobs.push_back(oob);
    rftc->trees.push_back(rftc->proto_tree);

  }

}


bool random_forest::multi_threaded_train(const ml_data &mld, 
					 ml_vector<rf_oob_indices> &oobs, 
					 ml_vector<dt_feature_importance> &forest_feature_importance) {

  ml_vector<std::thread> work_threads;
  ml_vector<rf_thread_config_ptr> thread_configs;

  // init thread input (# trees to build, custom seed, etc) and spawn the threads
  for(ml_uint thread_index = 0; thread_index < number_of_threads_; ++thread_index) {
    
    ml_uint ntrees = (number_of_trees_ / number_of_threads_);
    ntrees += (thread_index == 0) ? (number_of_trees_ % number_of_threads_) : 0;

    decision_tree proto_tree{mlid_, index_of_feature_to_predict_, 
	max_tree_depth_, min_leaf_instances_, 
        features_to_consider_per_node_, seed_ + thread_index};

    proto_tree.set_name(string_format("[thread %d]", thread_index));

    auto rftc = std::make_shared<rf_thread_config>(thread_index, ntrees, proto_tree, mld);
    thread_configs.push_back(rftc);
    work_threads.emplace_back(std::thread([rftc] { multi_threaded_work(rftc); }));
  }


  // wait for all threads to finish
  for(auto &thread : work_threads) {
    thread.join();
  }

						
  // combine trees, out of bag maps, and feature importance
  for(auto &thread_config : thread_configs) {

    if(thread_config->trees.size() != thread_config->number_of_trees) {
      log_error("some trees failed to build in thread %d\n", thread_config->thread_index);
      return(false);
    }

    for(ml_uint tree_index = 0; tree_index < thread_config->number_of_trees; ++tree_index) {
      oobs.push_back(thread_config->oobs[tree_index]);
      trees_.push_back(thread_config->trees[tree_index]);
      collect_feature_importance(thread_config->trees[tree_index].feature_importance(), forest_feature_importance);
    }

  }

  return(true);
}


bool random_forest::train(const ml_data &mld) {

  trees_.clear();
  feature_importance_.clear();
  oob_predictions_.clear();

  if(mlid_.empty()) {
    log_error("rf train() invalid instance definition...\n");
    return(false);
  }

  ml_vector<rf_oob_indices> oobs;
  ml_vector<dt_feature_importance> forest_feature_importance;
  forest_feature_importance.resize(mlid_.size());

  bool forest_was_built = (number_of_threads_ <= 1) ? single_threaded_train(mld, oobs, forest_feature_importance) :
    multi_threaded_train(mld, oobs, forest_feature_importance);

  if(!forest_was_built) {
    log_error("hit a snag while building the forest...\n");
    return(false);
  }

  feature_importance_ = calculate_feature_importance(mlid_, index_of_feature_to_predict_, forest_feature_importance);

  if(evaluate_oob_) {
    evaluate_out_of_bag(mld, oobs);
  }

  return(true);
}


ml_feature_value random_forest::evaluate(const ml_instance &instance) const {

  ml_feature_value rf_eval = {};

  if(trees_.empty()) {
    log_warn("evaluate() called on an empty forest\n");
    return(rf_eval);
  }

  ml_double sum = 0;
  ml_map<ml_uint, ml_uint> prediction_map;  

  //
  // evaluate all trees in the forest for the instance
  //
  for(const auto &tree : trees_) {
    ml_feature_value tree_eval = tree.evaluate(instance);
    if(type_ == ml_model_type::classification) {
      prediction_map[tree_eval.discrete_value_index] += 1;
    }
    else {
      sum += tree_eval.continuous_value;
    }
  }

  if(type_ == ml_model_type::classification) {
    //
    // classification returns the mode
    //
    ml_uint predicted_discrete_value_index = 0;
    ml_uint predicted_count = 0;
    // we iterate over all categories in a fixed order so that
    // ties in voting between categories are broken in a determinate
    // way (not dependent on the order of iteration within the container)
    ml_uint categories = mlid_[index_of_feature_to_predict_]->discrete_values.size();
    for(ml_uint category = 0; category < categories; ++category) {
      auto it = prediction_map.find(category);
      if(it == prediction_map.end()) {
	continue;
      }

      if(it->second > predicted_count) {
	predicted_discrete_value_index = it->first;
	predicted_count = it->second;
      }
    }

    rf_eval.discrete_value_index = predicted_discrete_value_index;
  }
  else {
    //
    // regression returns the mean
    //
    rf_eval.continuous_value = sum / trees_.size();
  }
  

  return(rf_eval);
}


bool random_forest::write_random_forest_base_info_to_file(const ml_string &path) const {

  json json_object = {{"object", "random_forest"},
		      {"version", ML_VERSION_STRING},
		      {"type", type_},
		      {"index_of_feature_to_predict", index_of_feature_to_predict_},
		      {"number_of_trees", number_of_trees_},
		      {"seed", seed_},
		      {"number_of_threads", number_of_threads_},
		      {"max_tree_depth", max_tree_depth_},
		      {"min_leaf_instances", min_leaf_instances_},
		      {"features_to_consider_per_node", features_to_consider_per_node_},
		      {"evaluate_oob", evaluate_oob_}};

  std::ofstream modelout(path);
  modelout << std::setw(4) << json_object << std::endl; 
  return(true);
}


bool random_forest::save(const ml_string &path) const {

  if(mlid_.empty()) {
    return(false);
  }

  if(!prepare_directory_for_model_save(path)) {
    return(false);
  }

  // write the instance definition
  if(!write_instance_definition_to_file(path + "/" + RF_MLID_FILE, mlid_)) {
    log_error("couldn't write rf instance definition to %s\n", RF_MLID_FILE.c_str());
    return(false);
  }

  // store the forest type, index to predict, etc
  if(!write_random_forest_base_info_to_file(path + "/" + RF_BASEINFO_FILE)) {
    log_error("couldn't write rf info to %s\n", RF_BASEINFO_FILE.c_str());
    return(false);
  }

  std::time_t timestamp = std::time(0); 

  // each tree is written to its own file
  for(std::size_t ii = 0; ii < trees_.size(); ++ii) {
    // we use the timestamp in the filename to make it easier to consolidate trees from multiple runs.
    // for example, tree1.1457973944.json
    ml_string filename = path + "/" + puml::TREE_MODEL_FILE_PREFIX + std::to_string(ii+1) + "." + std::to_string(timestamp) + ".json";
    if(!trees_[ii].save(filename, true)) {
      log_error("couldn't write tree to file: %s\n", filename.c_str());
      return(false);
    }
  }
 
  return(true);
  
}


bool random_forest::read_random_forest_base_info_from_file(const ml_string &path) {

  std::ifstream jsonfile(path);
  json json_object;
  jsonfile >> json_object;

  ml_string object_name = json_object["object"];
  if(object_name != "random_forest") {
    log_error("json object is not a random forest...\n");
    return(false);
  }

  if(!(get_modeltype_value_from_json(json_object, "type", type_) &&
       get_numeric_value_from_json(json_object, "index_of_feature_to_predict", index_of_feature_to_predict_) &&
       get_numeric_value_from_json(json_object, "number_of_trees", number_of_trees_) && 
       get_numeric_value_from_json(json_object, "seed", seed_) &&
       get_numeric_value_from_json(json_object, "number_of_threads", number_of_threads_) &&
       get_numeric_value_from_json(json_object, "max_tree_depth", max_tree_depth_) &&
       get_numeric_value_from_json(json_object, "min_leaf_instances", min_leaf_instances_) &&
       get_numeric_value_from_json(json_object, "features_to_consider_per_node", features_to_consider_per_node_) &&
       get_bool_value_from_json(json_object, "evaluate_oob", evaluate_oob_))) {
    return(false);
  }

  return(true);
}


bool random_forest::restore(const ml_string &path) {
  
  if(!read_instance_definition_from_file(path + "/" + RF_MLID_FILE, mlid_)) {
    log_error("couldn't read rf instance defintion\n");
    return(false);
  }

  if(!read_random_forest_base_info_from_file(path + "/" + RF_BASEINFO_FILE)) {
    log_error("couldn't read rf base info\n");
    return(false);
  }

  if(!read_decision_trees_from_directory(path, mlid_, trees_)) {
    return(false);
  }

  return(true);
}


ml_string random_forest::summary() const {

  if(mlid_.empty() || trees_.empty()) {
    return("(empty forest)\n");
  }

  ml_string desc;
  desc += "\n\n*** Random Forest Summary ***\n\n";
  desc += "Feature To Predict: " + mlid_[index_of_feature_to_predict_]->name + "\n";
  ml_string type_str = (type_ == ml_model_type::regression) ? "regression" : "classification";
  desc += "Type: " + type_str;
  desc += ", Trees: " + std::to_string(number_of_trees_);
  desc += ", Threads: " + std::to_string(number_of_threads_);
  desc += ", Max Depth: " + std::to_string(max_tree_depth_);
  desc += ", Min Leaf Instances: " + std::to_string(min_leaf_instances_);
  desc += ", Features p/n: " + std::to_string(features_to_consider_per_node_);
  desc += ", Seed: " + std::to_string(seed_);
  desc += ", Eval Out-Of-Bag: " + std::to_string(evaluate_oob_);
  desc += "\n";
  desc += feature_importance_summary(); 

  return(desc);
}


ml_string random_forest::feature_importance_summary() const {

  if(feature_importance_.empty()) {
    return("");
  }

  ml_string desc = "\n*** Feature Importance ***\n\n";
  for(const auto &importance_tuple : feature_importance_) {
    desc += std::get<1>(importance_tuple) + "\n";
  }

  return(desc);
}
 

} // namespace puml
