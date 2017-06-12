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
#include "brent/brent.h"

#include <sstream>
#include <string.h>

namespace puml {

static const ml_string &BOOSTED_BASEINFO_FILE = "boosted.json";
static const ml_string &BOOSTED_MLID_FILE = "mlid.json";


boosted_trees::boosted_trees(const ml_instance_definition &mlid,
			     const ml_string &feature_to_predict,
			     ml_uint number_of_trees,
			     ml_float learning_rate,
			     ml_uint seed,
			     ml_uint max_tree_depth,
			     ml_float subsample,
			     ml_uint min_leaf_instances,
			     ml_uint features_to_consider) :
  mlid_(mlid),
  index_of_feature_to_predict_(index_of_feature_with_name(feature_to_predict, mlid)),
  number_of_trees_(number_of_trees),
  learning_rate_(learning_rate),
  seed_(seed),
  max_tree_depth_(max_tree_depth),
  subsample_(subsample),
  min_leaf_instances_(min_leaf_instances),
  features_to_consider_per_node_(features_to_consider) {

  type_ = (mlid_[index_of_feature_to_predict_]->type == ml_feature_type::discrete) ? ml_model_type::classification : ml_model_type::regression;
  
  if(features_to_consider_per_node_ == BT_DEFAULT_FEATURES_HALF) {
    features_to_consider_per_node_ = (ml_uint) (((mlid_.size() - 1) / 2.0) + 0.5);
  }

  if(subsample_ < 0.001) {
    subsample_ = 0.5;
  }
}


static void sample_without_replacement(const ml_data &mld, ml_data &mld_iter, ml_float subsample, ml_rng &rng) {
  mld_iter.clear();

  ml_uint thresh = subsample * 10000.0;
  thresh = (thresh == 0) ? 5000 : thresh;

  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    if((rng.random_number() % 10000) < thresh) {
      mld_iter.push_back(mld[ii]);
    }
  }
}


struct leaf_opt_helper {

  ml_data *instances;
  const ml_instance_definition *mlid;
  boosted_loss_func loss_func; 

};


static double leaf_optimization(double x, void *user) {

  leaf_opt_helper *help = (leaf_opt_helper *) user;

  double loss = 0.0;

  for(const auto &inst_ptr : *help->instances) {
    const ml_instance &instance = *inst_ptr;
    double yi = instance[help->mlid->size()].continuous_value;
    double yhat = instance[help->mlid->size() + 1].continuous_value + x;

    if(help->loss_func) {
      loss += help->loss_func(yi, yhat);
    }
  }

  return(loss);
}


static void _gather_leaf_nodes(ml_vector<dt_node_ptr> &leaf_nodes, dt_node_ptr &node) {
  if(node->node_type == dt_node_type::leaf) {
    leaf_nodes.push_back(node);
    return;
  }

  if(node->split_left_node) {
    _gather_leaf_nodes(leaf_nodes, node->split_left_node);
  }

  if(node->split_right_node) {
    _gather_leaf_nodes(leaf_nodes, node->split_right_node);
  }
}


static void gather_leaf_nodes(ml_vector<dt_node_ptr> &leaf_nodes, decision_tree &tree) {

  leaf_nodes.clear();
  _gather_leaf_nodes(leaf_nodes, tree.root());
}


static void optimize_leaf_nodes(const ml_instance_definition &mlid, boosted_loss_func loss_func, decision_tree &tree) {

  ml_vector<dt_node_ptr> leaf_nodes;
  gather_leaf_nodes(leaf_nodes, tree);

  double eps = sqrt(r8_epsilon());

  for(auto &leaf_ptr : leaf_nodes) {

    //
    // use custom loss function if given, otherwise defaults to squared error loss
    //
    if(loss_func) {
      leaf_opt_helper help;
      help.instances = &leaf_ptr->leaf_instances;
      help.mlid = &mlid;
      help.loss_func = loss_func;
      
      double optimal = 0.0;
      double upper = leaf_ptr->feature_value.continuous_value * 100.0;
      double lower = upper * -1.0;
      if(lower > upper) {
	std::swap(lower, upper);
      }
      
      //puml::log("optimize: leaf before %.3f, ", leaf_ptr->feature_value.continuous_value);
      local_min(lower, upper, eps, eps, leaf_optimization, &help, &optimal);
      leaf_ptr->feature_value.continuous_value = optimal;
      //puml::log(" after %.3f\n", leaf_ptr->feature_value.continuous_value);
    }

    // empty the leaf instances vector. we only kept them around for this optimization step
    leaf_ptr->leaf_instances.clear();
    leaf_ptr->leaf_instances.shrink_to_fit();
  }
}


bool boosted_trees::train(const ml_data &mld) {

  trees_.clear();
  
  if(type_ != ml_model_type::regression) {
    log_error("boosting only implemented for regression...\n");
    return(false);
  }

  //
  // store the original target value as an unused
  // feature at the end of each instance, and
  // the boosted ensemble prediction after that.
  //
  for(const auto &inst_ptr : mld) {
    ml_instance &instance = *inst_ptr;
    ml_feature_value &fv = instance[index_of_feature_to_predict_];
    instance.push_back(fv);

    ml_feature_value ensemble = {};
    instance.push_back(ensemble);
  }


  decision_tree boosted_tree{mlid_, index_of_feature_to_predict_,
      max_tree_depth_, min_leaf_instances_, 
      features_to_consider_per_node_, seed_, true};

  ml_rng rng(seed_);

  //
  // build the ensemble. early stopping via callback
  //
  for(ml_uint ii=0; ii < number_of_trees_; ++ii) {

    ml_data mld_iter;   
    sample_without_replacement(mld, mld_iter, subsample_, rng);
    boosted_tree.set_seed(seed_ + ii);

    //
    // start with the optimal constant model
    //
    boosted_tree.set_max_tree_depth((ii == 0) ? 0 : max_tree_depth_);

    log("\nbuilding boosted tree %u\n", ii+1);
    if(!boosted_tree.train(mld_iter)) {
      log_error("failed to build boosted tree...\n");
      return(false);
    }

    optimize_leaf_nodes(mlid_, loss_func_, boosted_tree);
    trees_.push_back(boosted_tree);

    //
    // update the residual
    //
    for(const auto &inst_ptr : mld) {

      ml_instance &instance = *inst_ptr;
      ml_feature_value &residual = instance[index_of_feature_to_predict_];
      ml_double yi = instance[mlid_.size()].continuous_value;

      ml_feature_value pred = boosted_tree.evaluate(instance);
      instance[mlid_.size() + 1].continuous_value += (ii == 0) ? pred.continuous_value : (learning_rate_ * pred.continuous_value);
      ml_double yhat = instance[mlid_.size() + 1].continuous_value;

      //
      // use custom gradient function if given, otherwise squared error gradient
      //
      if(gradient_func_) {
	residual.continuous_value = gradient_func_(yi, yhat);
      }
      else {
	residual.continuous_value = yi - yhat;
      }
     
    }

    //
    // if a progress callback was given, exercise
    // and make this the final boosting iteration if 
    // it returns false
    //
    if(progress_callback_) {
      if(!progress_callback_(ii+1)) {
	break;
      }
    }

  }


  //
  // restore the original target value, and
  // remove the temp features added to the end 
  // of each instance
  //
  for(const auto &inst_ptr : mld) {
    ml_instance &instance = *inst_ptr;
    instance[index_of_feature_to_predict_] = instance[mlid_.size()];
    instance.resize(mlid_.size());
  }


  return(true);
}


ml_feature_value boosted_trees::evaluate(const ml_instance &instance) const {

  ml_feature_value result = {};
  if(trees_.empty()) {
    log_warn("evaluate() called on an empty boosted trees ensemble...\n");
    return(result);
  }

  ml_double ensemble_prediction = 0;
  for(std::size_t bb = 0; bb < trees_.size(); ++bb) {
    ml_feature_value tree_prediction = trees_[bb].evaluate(instance);
    if(bb == 0) { // first tree in the ensemble is the optimal constant model, a single terminal node
      ensemble_prediction += tree_prediction.continuous_value;
    }
    else {
      ensemble_prediction += (learning_rate_ * tree_prediction.continuous_value);
    }
  }

  result.continuous_value = ensemble_prediction;

  return(result);
}


bool boosted_trees::read_boosted_trees_base_info_from_file(const ml_string &path) {

  cJSON *json_object = read_model_json_from_file(path);
  if(!json_object) {
    log_error("couldn't load boosted trees json object from model file: %s\n", path.c_str());
    return(false);
  }

  cJSON *object = cJSON_GetObjectItem(json_object, "object");
  if(!object || !object->valuestring || strcmp(object->valuestring, "boosted_trees")) {
    log_error("json object is not a boosted tree ensemble...\n");
    return(false);
  }

  if(!(get_modeltype_value_from_json(json_object, "type", type_) &&
       get_numeric_value_from_json(json_object, "index_of_feature_to_predict", index_of_feature_to_predict_) &&
       get_numeric_value_from_json(json_object, "number_of_trees", number_of_trees_) && 
       get_float_value_from_json(json_object, "learning_rate", learning_rate_) &&
       get_numeric_value_from_json(json_object, "seed", seed_) &&
       get_numeric_value_from_json(json_object, "max_tree_depth", max_tree_depth_) &&
       get_float_value_from_json(json_object, "subsample", subsample_) &&
       get_numeric_value_from_json(json_object, "min_leaf_instances", min_leaf_instances_) &&
       get_numeric_value_from_json(json_object, "features_to_consider_per_node", features_to_consider_per_node_))) {
    return(false);
  }

  cJSON_Delete(json_object);

  return(true);
}


bool boosted_trees::restore(const ml_string &path) {

  if(!read_instance_definition_from_file(path + "/" + BOOSTED_MLID_FILE, mlid_)) {
    log_error("couldn't read boosted trees instance defintion\n");
    return(false);
  }

  if(!read_boosted_trees_base_info_from_file(path + "/" + BOOSTED_BASEINFO_FILE)) {
    log_error("couldn't read boosted trees base info\n");
    return(false);
  }

  if(!read_decision_trees_from_directory(path, mlid_, trees_)) {
    return(false);
  }

  return(true);
}


bool boosted_trees::write_boosted_trees_base_info_to_file(const ml_string &path) const {
  
  cJSON *json_boosted = cJSON_CreateObject();
  if(!json_boosted) {
    log_error("couldn't create json object from boosted trees\n");
    return(false);
  }

  cJSON_AddStringToObject(json_boosted, "object", "boosted_trees");
  cJSON_AddNumberToObject(json_boosted, "type", (double)type_);
  cJSON_AddNumberToObject(json_boosted, "index_of_feature_to_predict", index_of_feature_to_predict_);
  cJSON_AddNumberToObject(json_boosted, "number_of_trees", number_of_trees_);
  cJSON_AddNumberToObject(json_boosted, "learning_rate", learning_rate_);
  cJSON_AddNumberToObject(json_boosted, "seed", seed_);
  cJSON_AddNumberToObject(json_boosted, "max_tree_depth", max_tree_depth_);
  cJSON_AddNumberToObject(json_boosted, "subsample", subsample_);
  cJSON_AddNumberToObject(json_boosted, "min_leaf_instances", min_leaf_instances_);
  cJSON_AddNumberToObject(json_boosted, "features_to_consider_per_node", features_to_consider_per_node_);

  bool status = write_model_json_to_file(path, json_boosted);
  cJSON_Delete(json_boosted);
  
  return(status);
}


bool boosted_trees::save(const ml_string &path) const {

  if(mlid_.empty()) {
    return(false);
  }

  if(!prepare_directory_for_model_save(path)) {
    return(false);
  }

  // write the instance definition
  if(!write_instance_definition_to_file(path + "/" + BOOSTED_MLID_FILE, mlid_)) {
    log_error("couldn't write boosted instance definition to %s\n", BOOSTED_MLID_FILE.c_str());
    return(false);
  }

  // store the tree type, index to predict, and learning rate
  if(!write_boosted_trees_base_info_to_file(path + "/" + BOOSTED_BASEINFO_FILE)) {
    log_error("couldn't write boosted info to %s\n", BOOSTED_BASEINFO_FILE.c_str());
    return(false);
  }

  // each tree in the ensemble is written to its own file
  for(std::size_t ii = 0; ii < trees_.size(); ++ii) {
    ml_string filename =  path + "/" + puml::TREE_MODEL_FILE_PREFIX + std::to_string(ii+1) + ".json";
    if(!trees_[ii].save(filename, true)) {
      log_error("couldn't write boosted tree to file: %s\n", filename.c_str());
      return(false);
    }
  }

  return(true);
}


ml_string boosted_trees::summary() const {

  if(mlid_.empty() || trees_.empty()) {
    return("(empty ensemble)\n");
  }

  ml_string desc;
  desc += "\n\n*** Boosted Trees Summary ***\n\n";
  desc += "Feature To Predict: " + mlid_[index_of_feature_to_predict_]->name + "\n";
  ml_string type_str = (type_ == ml_model_type::regression) ? "regression" : "classification";
  desc += "Type: " + type_str;
  desc += ", Trees: " + std::to_string(number_of_trees_);
  desc += ", Max Depth: " + std::to_string(max_tree_depth_);
  desc += ", Learning Rate: " + std::to_string(learning_rate_);
  desc += ", Subsample: " + std::to_string(subsample_);
  desc += ", Min Leaf Instances: " + std::to_string(min_leaf_instances_);
  desc += ", Features p/n: " + std::to_string(features_to_consider_per_node_);
  desc += ", Seed: " + std::to_string(seed_);
  desc += "\n";

  return(desc);
}


} // namespace puml
