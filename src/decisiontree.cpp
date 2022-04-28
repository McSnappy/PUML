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

#include <fstream>
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>

#include "decisiontree.h"
#include "mlutil.h"

namespace puml {

static const ml_string &DT_TREE_JSONFILE = "tree.json";
static const ml_string &DT_MLID_JSONFILE = "mlid.json";
static const ml_double DT_COMPARISON_EQUAL_TOL = 0.00000001;

struct dt_split {
  ml_uint split_feature_index;
  ml_feature_type split_feature_type;
  ml_feature_value split_feature_value;
  dt_comparison_op split_left_op = dt_comparison_op::noop;
  dt_comparison_op split_right_op = dt_comparison_op::noop;
  ml_double left_score = 0;
  ml_double right_score = 0;
};


decision_tree::decision_tree(const ml_instance_definition &mlid, 
			     const ml_string &feature_to_predict, 
			     ml_uint max_tree_depth, ml_uint min_leaf_instances, 
			     ml_uint features_to_consider_per_node, 
			     ml_uint seed,
			     bool keep_instances_at_leaf_nodes) :
  mlid_(mlid),
  index_of_feature_to_predict_(index_of_feature_with_name(feature_to_predict, mlid)),
  max_tree_depth_(max_tree_depth),
  min_leaf_instances_(min_leaf_instances),
  features_to_consider_per_node_(features_to_consider_per_node),
  seed_(seed),
  keep_instances_at_leaf_nodes_(keep_instances_at_leaf_nodes),
  rng_(seed_) {
  type_ = (mlid_[index_of_feature_to_predict_]->type == ml_feature_type::discrete) ? ml_model_type::classification : ml_model_type::regression;
}


decision_tree::decision_tree(const ml_instance_definition &mlid, 
			     ml_uint index_of_feature_to_predict, 
			     ml_uint max_tree_depth, ml_uint min_leaf_instances, 
			     ml_uint features_to_consider_per_node, 
			     ml_uint seed,
			     bool keep_instances_at_leaf_nodes) :
  mlid_(mlid),
  index_of_feature_to_predict_(index_of_feature_to_predict),
  max_tree_depth_(max_tree_depth),
  min_leaf_instances_(min_leaf_instances),
  features_to_consider_per_node_(features_to_consider_per_node),
  seed_(seed),
  keep_instances_at_leaf_nodes_(keep_instances_at_leaf_nodes),
  rng_(seed_) {
  type_ = (mlid_[index_of_feature_to_predict_]->type == ml_feature_type::discrete) ? ml_model_type::classification : ml_model_type::regression;
}


bool decision_tree::validate_for_training(const ml_data &mld) {

  if(mlid_.empty()) {
    log_error("empty instance definition...\n");
    return(false);
  }

  if(mld.empty()) {
    log_error("empty instance data set...\n");
    return(false);
  }

  if(mld[0]->size() < mlid_.size()) {
    log_error("feature count mismatch b/t instance definition and instance data\n");
    return(false);
  }

  if(index_of_feature_to_predict_ >= mlid_.size()) {
    log_error("invalid index of feature to predict...\n");
    return(false);
  }

  if(min_leaf_instances_ == 0) {
    log_error("minimum leaf instances must be greater than 0\n");
    return(false);
  }

  return(true);  
}


static ml_double calc_mean_for_continuous_feature(ml_uint feature_index, const ml_data &mld) {

  if(mld.empty()) {
    return(0.0);
  }

  ml_double sum = 0.0;
  for(const auto &inst_ptr : mld) {
    sum += (*inst_ptr)[feature_index].continuous_value;
  }

  ml_double mean = sum / mld.size();
  return(mean);
}


static ml_uint calc_mode_value_index_for_discrete_feature(ml_uint feature_index, const ml_data &mld) {

  std::map<ml_uint, ml_uint> value_map;

  for(const auto &inst_ptr : mld) {
    ml_uint discrete_value_index = (*inst_ptr)[feature_index].discrete_value_index;
    value_map[discrete_value_index] += 1;
  }

  ml_uint mindex = 0, mmax = 0;
  for(auto it = value_map.begin(); it != value_map.end(); ++it) {
    if(it->second > mmax) {
      mindex = it->first;
      mmax = it->second;
    }
  }

  return(mindex);
}


static bool continuous_feature_satisfies_constraint(const ml_feature_value &feature_value, const ml_feature_value &split_feature_value, dt_comparison_op op) {
  switch(op) {
  case dt_comparison_op::lessthanorequal: return((feature_value.continuous_value < split_feature_value.continuous_value)); break; // OREQUAL, ha...
  case dt_comparison_op::greaterthan: return((feature_value.continuous_value > split_feature_value.continuous_value)); break; 
  default: log_error("confused by invalid split comparison operator %d (cont)... exiting.\n", op); exit(1); break;
  }

  return(false);
}


static bool discrete_feature_satisfies_constraint(const ml_feature_value &feature_value, const ml_feature_value &split_feature_value, dt_comparison_op op) {
  switch(op) {
  case dt_comparison_op::equal: return((feature_value.discrete_value_index == split_feature_value.discrete_value_index)); break;
  case dt_comparison_op::notequal: return((feature_value.discrete_value_index != split_feature_value.discrete_value_index)); break;
  default: log_error("confused by invalid split comparison operator %d (disc)... exiting.\n", op); exit(1); break;
  }

  return(false);
}


static bool instance_satisfies_constraint_of_split(const ml_instance &instance, ml_uint split_feature_index, ml_feature_type split_feature_type, 
						   const ml_feature_value &split_feature_value, dt_comparison_op split_op) {
 
  if(split_feature_index >= instance.size()) {
    log_error("invalid split feature index: %d. exiting\n", split_feature_index); 
    exit(1);
  }
   
  const ml_feature_value &mlfv = instance[split_feature_index];

  switch(split_feature_type) {
  case ml_feature_type::continuous: return(continuous_feature_satisfies_constraint(mlfv, split_feature_value, split_op)); break;
  case ml_feature_type::discrete: return(discrete_feature_satisfies_constraint(mlfv, split_feature_value, split_op)); break;
  default: log_error("confused by split feature type... exiting.\n"); exit(1); break;
  }
    
  return(false);
}


static void perform_split(const ml_data &mld, const dt_split &split, ml_data &leftmld, ml_data &rightmld) {
 
  leftmld.reserve(mld.size());
  rightmld.reserve(mld.size());

  for(const auto &inst_ptr : mld) {
    if(instance_satisfies_constraint_of_split(*inst_ptr, split.split_feature_index, split.split_feature_type, split.split_feature_value, split.split_left_op)) {
      leftmld.push_back(inst_ptr);
    }
    else {
      rightmld.push_back(inst_ptr);
    }
  }
}


static void add_splits_for_discrete_feature(ml_uint feature_index, const ml_data &mld, ml_vector<dt_split> &splits) {
  
  if(mld.empty()) {
    return;
  }

  ml_set<ml_uint> levels;
  for(const auto &inst_ptr : mld) {
    ml_uint level = (*inst_ptr)[feature_index].discrete_value_index;
    levels.insert(level);
  }

  //
  // only 1 level so no split possible
  //
  if(levels.size() == 1) {
    return;
  }

  // 
  // if only 2 levels are present remove one since checking both is redundant 
  //
  if(levels.size() == 2) {
    levels.erase(levels.begin());
  }

  for(const ml_uint &level : levels) {
    dt_split dsplit{};
    dsplit.split_feature_index = feature_index;
    dsplit.split_feature_type = ml_feature_type::discrete;
    dsplit.split_feature_value.discrete_value_index = level;
    dsplit.split_right_op = dt_comparison_op::equal;
    dsplit.split_left_op = dt_comparison_op::notequal;
    splits.push_back(dsplit);
  }
  
}


static void add_splits_for_continuous_feature(ml_uint feature_index, const ml_data &mld, ml_vector<dt_split> &splits) {

  // 
  // add splits based on the distribution of the feature in mld
  //

  if(mld.empty()) {
    return;
  }

  ml_uint count = 0;
  ml_double mean = 0, M2 = 0;

  for(const auto &inst_ptr : mld) {
    ml_double fval = (*inst_ptr)[feature_index].continuous_value;
    ++count;
    ml_double delta = fval - mean;
    mean = mean + (delta / count);
    M2 = M2 + (delta * (fval - mean));
  }

  ml_double std = (count < 2) ? 0.0 : std::sqrt( M2 / (count - 1));

  dt_split csplit{};
  csplit.split_feature_index = feature_index;
  csplit.split_feature_type = ml_feature_type::continuous;
  csplit.split_right_op = dt_comparison_op::greaterthan;
  csplit.split_left_op = dt_comparison_op::lessthanorequal;
  
  csplit.split_feature_value.continuous_value = mean;
  splits.push_back(csplit);    

  if(std > 0.0) {
    csplit.split_feature_value.continuous_value = mean + (std / 2.0);
    splits.push_back(csplit);  

    csplit.split_feature_value.continuous_value = mean - (std / 2.0);
    splits.push_back(csplit);      
  }
  
}


//
// score regions for regression using residual sum of squares (approx)
// returns a tuple with (left region score, right region score, combined score)
//
static std::tuple<ml_double, ml_double, ml_double> score_regions_with_split_for_regression(const ml_data &mld, 
											   const dt_split &split, 
											   const decision_tree &tree) {

  ml_double lscore = 0, rscore = 0, combined_score = 0;
  ml_uint n_left = 0, n_right = 0;
  ml_double mean_left = 0, mean_right = 0;
  
  for(const auto &inst_ptr : mld) {
    
    ml_double feature_val = (*inst_ptr)[tree.index_of_feature_to_predict()].continuous_value;
    
    if((split.split_left_op == dt_comparison_op::noop) || 
       instance_satisfies_constraint_of_split(*inst_ptr, split.split_feature_index, 
					      split.split_feature_type, split.split_feature_value,
					      split.split_left_op)) {
      ++n_left;
      ml_double delta = feature_val - mean_left;
      mean_left = mean_left + (delta / n_left);
      lscore = lscore + (delta * (feature_val - mean_left));
    }
    else {
      ++n_right;
      ml_double delta = feature_val - mean_right;
      mean_right = mean_right + (delta / n_right);
      rscore = rscore + (delta * (feature_val - mean_right));
    }
    
  }

  combined_score = lscore + rscore;

  return(std::make_tuple(lscore, rscore, combined_score));
}


//
// score regions for classification using Gini index
// returns a tuple with (left region score, right region score, combined score)
//
static std::tuple<ml_double, ml_double, ml_double> score_regions_with_split_for_classification(const ml_data &mld, 
											       const dt_split &split, 
											       const decision_tree &tree) {

  ml_double lscore = 0, rscore = 0, combined_score = 0;
  ml_map<ml_uint, ml_uint> left_value_map;
  ml_map<ml_uint, ml_uint> right_value_map;
  ml_uint lcount=0, rcount=0;
  
  for(const auto &inst_ptr : mld) {
    
    ml_uint discrete_value_index = (*inst_ptr)[tree.index_of_feature_to_predict()].discrete_value_index;
    
    if((split.split_left_op == dt_comparison_op::noop) || 
       instance_satisfies_constraint_of_split(*inst_ptr, split.split_feature_index, 
					      split.split_feature_type, split.split_feature_value,
					      split.split_left_op)) {
      left_value_map[discrete_value_index] += 1;
      ++lcount;
    }
    else {
      right_value_map[discrete_value_index] += 1;
      ++rcount;
    }
    
  }
  
  for(auto it = left_value_map.begin(); it != left_value_map.end(); ++it) {
    ml_double class_proportion = ((ml_double) it->second / lcount);
    lscore += (class_proportion * (1.0 - class_proportion));
  }
  
  for(auto it = right_value_map.begin(); it != right_value_map.end(); ++it) {
    ml_double class_proportion = ((ml_double) it->second / rcount);
    rscore += (class_proportion * (1.0 - class_proportion));
  }
  
  combined_score = (((ml_double) lcount / mld.size()) * lscore) + (((ml_double) rcount / mld.size()) * rscore);
  return(std::make_tuple(lscore, rscore, combined_score));
}


//
// score the regions created by subdividing mld using split.
// returns a tuple with (left region score, right region score, combined score)
//
static std::tuple<ml_double, ml_double, ml_double> score_regions_with_split(const ml_data &mld, 
									    const dt_split &split, 
									    const decision_tree &tree) {

  if(mld.empty()) {
    return(std::make_tuple(0.0, 0.0, 0.0));
  }
  
  if(tree.type() == ml_model_type::regression) {
    return(score_regions_with_split_for_regression(mld, split, tree));
  }
  else {
    return(score_regions_with_split_for_classification(mld, split, tree));
  }

}


//
// score undivided mld using empty noop split
//
static ml_double score_region(const ml_data &mld, const decision_tree &tree) {
  ml_double lscore=0, rscore=0, combined_score=0;
  std::tie(lscore, rscore, combined_score) = score_regions_with_split(mld, dt_split{}, tree);
  return(combined_score);
}


static void pick_random_features_to_consider(const decision_tree &tree,
					     ml_rng &rng,
					     ml_map<ml_uint, bool> &random_features) {

  if(tree.features_to_consider_per_node() > (tree.mlid().size() - 1)) {
    log_warn("invalid random features config... considering all features.\n");
    return;
  }

  //log("considering random features of the set: ");

  while(random_features.size() < tree.features_to_consider_per_node()) {
    
    ml_uint feature_index = rng.random_number() % tree.mlid().size();

    if(feature_index == tree.index_of_feature_to_predict()) {
      continue;
    }

    if(random_features.find(feature_index) != random_features.end()) {
      continue;
    }

    //log("%u ", feature_index);

    random_features[feature_index] = true;
  }

  //log("\n");
}


bool decision_tree::find_best_split(const ml_data &mld, dt_split &best_split, ml_double score) {

  ml_vector<dt_split> splits;
  
  ml_map<ml_uint, bool> random_features_to_consider;
  if(features_to_consider_per_node_ > 0) {
    pick_random_features_to_consider(*this, rng_, random_features_to_consider);
  }

  for(std::size_t findex = 0; findex < mlid_.size(); ++findex) {

    if(findex == index_of_feature_to_predict_) {
      continue;
    }

    if((random_features_to_consider.size() > 0) && (random_features_to_consider.find(findex) == random_features_to_consider.end())) {
      continue;
    }

    switch(mlid_[findex]->type) {
    case ml_feature_type::discrete: add_splits_for_discrete_feature(findex, mld, splits); break;
    case ml_feature_type::continuous: add_splits_for_continuous_feature(findex, mld, splits); break;
    default: log_error("invalid feature type...\n"); break;
    }
  }

  //log("total splits to consider: %zu\n", splits.size());

  ml_double best_score,best_left_score,best_right_score;
  ml_uint best_split_index = 0;

  best_score = best_left_score = best_right_score = std::numeric_limits<ml_double>::max();

  for(std::size_t ii = 0; ii < splits.size(); ++ii) {
    
    ml_double lscore=0, rscore=0, combined_score=0;
    std::tie(lscore, rscore, combined_score) = score_regions_with_split(mld, splits[ii], *this);

    if(combined_score < best_score) {
      best_score = combined_score;
      best_left_score = lscore;
      best_right_score = rscore;
      best_split_index = ii;
    }
    
  }

  if(best_split_index < splits.size()) {
    best_split = splits[best_split_index];
    best_split.left_score = best_left_score;
    best_split.right_score = best_right_score;

    feature_importance_[best_split.split_feature_index].sum_score_delta += (score - best_score);
    feature_importance_[best_split.split_feature_index].count += 1;

    return(true);
  }

  best_split.split_feature_index = 0;
  best_split.split_left_op = best_split.split_right_op = dt_comparison_op::noop;
  return(false);
}


void decision_tree::config_leaf_node(const ml_data &mld, dt_node_ptr &leaf) {
  leaves_ += 1;
  leaf->node_type = dt_node_type::leaf;
  leaf->feature_index = index_of_feature_to_predict_;
  leaf->feature_type = mlid_[index_of_feature_to_predict_]->type;
  if(type_ == ml_model_type::regression) {
    leaf->feature_value.continuous_value = calc_mean_for_continuous_feature(leaf->feature_index, mld);
  }
  else {
    leaf->feature_value.discrete_value_index = calc_mode_value_index_for_discrete_feature(leaf->feature_index, mld);
  }

  if(keep_instances_at_leaf_nodes_) {
    leaf->leaf_instances = mld;
  }
}


static void config_split_node(const dt_split &split, dt_node_ptr &split_node) {
  split_node->node_type = dt_node_type::split;
  split_node->feature_index = split.split_feature_index;
  split_node->feature_type = split.split_feature_type;
  split_node->feature_value = split.split_feature_value;
  split_node->split_left_op = split.split_left_op;
  split_node->split_right_op = split.split_right_op;
}


bool decision_tree::prune_twin_leaf_nodes(dt_node_ptr &node) {

  // 
  // we prune sibling leaf nodes that predict the same class/value and
  // convert their parent from split to leaf node.
  //
 
  if((node->split_left_node->node_type == dt_node_type::leaf) && (node->split_right_node->node_type == dt_node_type::leaf)) {

    if(((type_ == ml_model_type::classification) && (node->split_left_node->feature_value.discrete_value_index == node->split_right_node->feature_value.discrete_value_index)) ||
       ((type_ == ml_model_type::regression) && (fabs(node->split_left_node->feature_value.continuous_value - node->split_right_node->feature_value.continuous_value) < DT_COMPARISON_EQUAL_TOL))) {
      nodes_ -= 2;
      leaves_ -= 2;
      node->split_left_node = nullptr;
      node->split_right_node = nullptr;
      return(true);
    }
  }

  return(false);
}


void decision_tree::build_tree_node(const ml_data &mld, dt_node_ptr &node, ml_uint depth, ml_double score) {
  
  node = std::make_shared<dt_node>();
  if(!node) {
    log_error("out of memory. aborting...\n");
    abort();
  }

  nodes_ += 1;

  if(depth == max_tree_depth_) {
    config_leaf_node(mld, node);
    return;
  }
  
  dt_split best_split = {};
  ml_data leftmld, rightmld;

  if(find_best_split(mld, best_split, score)) {
    perform_split(mld, best_split, leftmld, rightmld);
  }

  if((leftmld.size() < min_leaf_instances_) || 
     (rightmld.size() < min_leaf_instances_)) {
    config_leaf_node(mld, node);
    return;
  }

  config_split_node(best_split, node);

  build_tree_node(leftmld, node->split_left_node, depth+1, best_split.left_score);
  build_tree_node(rightmld, node->split_right_node, depth+1, best_split.right_score);

  if(prune_twin_leaf_nodes(node)) {
    config_leaf_node(mld, node);
  }
 
}


bool decision_tree::train(const ml_data &mld) {

  if(!validate_for_training(mld)) {
    return(false);
  }

  root_ = nullptr;
  nodes_ = leaves_ = 0;
  feature_importance_.clear();
  feature_importance_.resize(mlid_.size());

  auto t1 = std::chrono::high_resolution_clock::now();
  build_tree_node(mld, root_, 0, score_region(mld, *this)); 
  auto t2 = std::chrono::high_resolution_clock::now();
   
  ml_uint ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  log("built tree %s in %.3f seconds (%u leaves, %u nodes)\n", name_.c_str(), (ms / 1000.0), leaves_, nodes_); 

  return(true);
}


static ml_string name_for_split_operator(dt_comparison_op op) {
  
  ml_string op_name;

  switch(op) {
  case dt_comparison_op::noop : op_name = "no-op"; break;
  case dt_comparison_op::lessthanorequal: op_name = "<="; break;
  case dt_comparison_op::greaterthan: op_name = ">"; break;
  case dt_comparison_op::equal: op_name = "="; break;
  case dt_comparison_op::notequal: op_name = "!="; break;
  default: log_error("invalid split operator"); break;
  }

  return(op_name);
}


static void decision_tree_node_desc(const ml_instance_definition &mlid, const dt_node &node, ml_uint depth, ml_string &desc) {

  ml_string feature_value_as_string;
  if(node.feature_type == ml_feature_type::discrete) {
    feature_value_as_string = mlid[node.feature_index]->discrete_values[node.feature_value.discrete_value_index];
  }
  else {
    feature_value_as_string = std::to_string(node.feature_value.continuous_value);
  }

  if(node.node_type == dt_node_type::split) { 
    // Left Side
    desc += "\n";
    for(ml_uint ii=0; ii < depth; ++ii) {
      desc += "|  ";
    }

    desc += mlid[node.feature_index]->name + " ";
    desc += name_for_split_operator(node.split_left_op) + " ";
    desc += feature_value_as_string;
    decision_tree_node_desc(mlid, *node.split_left_node, depth+1, desc);
  
    // Right Side
    desc += "\n";
    for(ml_uint ii=0; ii < depth; ++ii) {
      desc += "|  ";
    }

    desc += mlid[node.feature_index]->name + " ";
    desc += name_for_split_operator(node.split_right_op) + " ";
    desc += feature_value_as_string;
    decision_tree_node_desc(mlid, *node.split_right_node, depth+1, desc);
  }
  else {
    // dt_node_type::leaf
    desc += ": " + feature_value_as_string;
  }
  
}
  

ml_string decision_tree::summary() const {

  if(mlid_.empty() || !root_) {
    return("(empty decision tree)\n");
  }

  ml_string desc;
  desc += "\n\n*** Decision Tree Summary ***\n\n";
  desc += "Feature To Predict: " + mlid_[index_of_feature_to_predict_]->name + "\n";
  ml_string type_str = (type_ == ml_model_type::regression) ? "regression" : "classification";
  desc += "Type: " + type_str;
  desc += ", Max Depth: " + std::to_string(max_tree_depth_);
  desc += ", Min Leaf Instances: " + std::to_string(min_leaf_instances_);
  if(features_to_consider_per_node_ > 0) {
    desc += ", Features p/n: " + std::to_string(features_to_consider_per_node_);
    desc += ", Seed: " + std::to_string(seed_);
  }
  
  desc += ", Leaves: " + std::to_string(leaves_);
  desc += ", Size: " + std::to_string(nodes_) + "\n";
  decision_tree_node_desc(mlid_, *root_, 0, desc);
  desc += "\n\n";

  return(desc);
}


static const ml_feature_value evaluate_decision_tree_node_for_instance(const dt_node &node, const ml_instance &instance) {

  if(node.node_type == dt_node_type::leaf) {
    return(node.feature_value);
  }

  if(instance_satisfies_constraint_of_split(instance, node.feature_index, node.feature_type, node.feature_value, node.split_left_op)) {
    return(evaluate_decision_tree_node_for_instance(*node.split_left_node, instance));
  }

  return(evaluate_decision_tree_node_for_instance(*node.split_right_node, instance));
}


ml_feature_value decision_tree::evaluate(const ml_instance &instance) const {

  ml_feature_value empty = {};
  if(!root_ || mlid_.empty()) {
    log_warn("evaluate called on an empty tree...\n");
    return(empty);
  }

  if(instance.size() < mlid_.size()) {
    log_error("feature count mismatch b/t instance definition and instance to evaluate\n");
    return(empty);
  }

  return(evaluate_decision_tree_node_for_instance(*root_, instance));
}

  
void add_nodes_to_json_object(const dt_node &node, json &json_nodes, ml_uint &node_id) {

  json anode = {{"id", node_id},
		{"nt", (double) node.node_type},
		{"fi", node.feature_index},
		{"ft", (double) node.feature_type},
		{"fv", (node.feature_type == ml_feature_type::continuous) ? node.feature_value.continuous_value : node.feature_value.discrete_value_index}};
  
  if(node.split_left_node) {
    ml_uint left_node_id = ++node_id;
    anode["lid"] = left_node_id;
    anode["lop"] = (ml_uint) node.split_left_op;
    add_nodes_to_json_object(*node.split_left_node, json_nodes, node_id); 
  }

  if(node.split_right_node) {
    ml_uint right_node_id = ++node_id;
    anode["rid"] = right_node_id;
    anode["rop"] = (ml_uint) node.split_right_op;
    add_nodes_to_json_object(*node.split_right_node, json_nodes, node_id);
  }

  json_nodes.push_back(anode);

}


static bool create_tree_node_from_json(dt_node_ptr &node, ml_uint node_id, ml_map<ml_uint, json> &nodes_map, ml_uint &nodes, ml_uint &leaves) {

  if(nodes_map.find(node_id) == nodes_map.end()) {
    log_error("can't find node in json with node_id of %d\n", node_id);
    return(false);
  }

  const json &json_node = nodes_map[node_id];
  
  node = std::make_shared<dt_node>();
  if(!node) {
    log_error("out of memory...\n");
    return(false);
  }

  nodes += 1;

  ml_uint node_type=0, feature_index=0, feature_type=0;
  if(!get_numeric_value_from_json(json_node, "nt", node_type) ||
     !get_numeric_value_from_json(json_node, "fi", feature_index) ||
     !get_numeric_value_from_json(json_node, "ft", feature_type)) {
    log_error("invalid or incomplete node json. node id: %u\n", node_id);
    return(false);
  }
     
  node->node_type = (dt_node_type) node_type;
  node->feature_type = (ml_feature_type) feature_type;
  node->feature_index = feature_index;
  if(node->feature_type == ml_feature_type::continuous) {
    get_float_value_from_json(json_node, "fv", node->feature_value.continuous_value);
  }
  else {
    get_numeric_value_from_json(json_node, "fv", node->feature_value.discrete_value_index);
  }

  if(node->node_type == dt_node_type::leaf) {
    leaves += 1;
  }
  else {

    ml_uint left_node_id=0, left_node_op=0, right_node_id=0, right_node_op=0;
    if(!get_numeric_value_from_json(json_node, "lid", left_node_id) ||
       !get_numeric_value_from_json(json_node, "lop", left_node_op) ||
       !get_numeric_value_from_json(json_node, "rid", right_node_id) ||
       !get_numeric_value_from_json(json_node, "rop", right_node_op)) {
      log_error("incomplete node json. node id: %u\n", node_id);
      return(false);
    }

    node->split_left_op = (dt_comparison_op) left_node_op;
    node->split_right_op = (dt_comparison_op) right_node_op;

    if(!create_tree_node_from_json(node->split_left_node, left_node_id, nodes_map, nodes, leaves) ||
       !create_tree_node_from_json(node->split_right_node, right_node_id, nodes_map, nodes, leaves)) {
      return(false);
    }
  }
   

  return(true);
}


bool decision_tree::create_decision_tree_from_json(const json &json_object) {

  if(json_object.empty()) {
    return(false);
  }

  ml_string object_name = json_object["object"];
  if(object_name != "decision_tree") {
    log_error("tree json is malformed...\n");
    return(false);
  }

  if(!(get_modeltype_value_from_json(json_object, "type", type_) &&
       get_numeric_value_from_json(json_object, "index_of_feature_to_predict", index_of_feature_to_predict_) &&
       get_numeric_value_from_json(json_object, "max_tree_depth", max_tree_depth_) &&
       get_numeric_value_from_json(json_object, "min_leaf_instances", min_leaf_instances_) &&
       get_numeric_value_from_json(json_object, "features_to_consider_per_node", features_to_consider_per_node_) &&
       get_numeric_value_from_json(json_object, "seed", seed_) &&
       get_bool_value_from_json(json_object, "keep_instances_at_leaf_nodes", keep_instances_at_leaf_nodes_))) {
    return(false);
  }

  if(!json_object.contains<ml_string>("nodes")) {
    log_error("json object is missing a nodes array\n");
    return(false);
  }

  json nodes_array = json_object["nodes"];
  
  ml_map<ml_uint, json> nodes_map;
  for(const json &json_node : nodes_array) {

    ml_uint node_id = 0;
    if(!get_numeric_value_from_json(json_node, "id", node_id)) {
      log_error("tree json has node with missing node_id\n");
      return(false);
    }

    nodes_map[node_id] = json_node;
  }

  if(nodes_map.empty()) {
    log_error("tree json has empty nodes array\n");
    return(false);
  }

  if(!create_tree_node_from_json(root_, 0, nodes_map, nodes_, leaves_)) {
    log_error("failed to build tree nodes from json...\n");
    return(false);
  }

  return(true);
}


bool decision_tree::save(const ml_string &path, bool part_of_ensemble) const {

  if(mlid_.empty() || !root_) {
    return(false);
  }

  ml_uint node_id = 0;
  json json_nodes = json::array();
  add_nodes_to_json_object(*root_, json_nodes, node_id);
  
  json json_tree = {
    {"version", ML_VERSION_STRING},
    {"object", "decision_tree"},
    {"type", (double)type_},
    {"index_of_feature_to_predict", index_of_feature_to_predict_},
    {"max_tree_depth", max_tree_depth_},
    {"min_leaf_instances", min_leaf_instances_},
    {"features_to_consider_per_node", features_to_consider_per_node_},
    {"seed", seed_},
    {"keep_instances_at_leaf_nodes", keep_instances_at_leaf_nodes_},
    {"nodes", json_nodes}
  };

  if(part_of_ensemble) {
    //
    // a tree within an ensemble (random_forest) writes
    // just its tree json without the mlid, which is written by the 
    // ensemble model.
    //
    std::ofstream modelout(path);
    modelout << json_tree << std::endl; 
  }
  else {
    //
    // a single tree has its own directory with tree json and mlid json
    //
    if(prepare_directory_for_model_save(path)) {
      std::ofstream modelout(path + "/" + DT_TREE_JSONFILE);
      modelout << json_tree << std::endl; 
      write_instance_definition_to_file(path + "/" + DT_MLID_JSONFILE, mlid_);
    }
  }

  return(true);
}


bool decision_tree::restore(const ml_string &path) {
  ml_instance_definition mlid;
  if(!read_instance_definition_from_file(path + "/" + DT_MLID_JSONFILE, mlid)) {
    return(false);
  }

  return(restore(path + "/" + DT_TREE_JSONFILE, mlid));
}


bool decision_tree::restore(const ml_string &path, const ml_instance_definition &mlid) {

  std::ifstream jsonfile(path);
  json json_object;
  jsonfile >> json_object;

  mlid_ = mlid;
  root_ = nullptr;
  leaves_ = nodes_ = 0;
  feature_importance_.clear();
  bool status = create_decision_tree_from_json(json_object);
  
  return(status);  
}


} // namespace puml
