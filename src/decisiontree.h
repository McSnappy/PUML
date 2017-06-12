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

#pragma once

#include <iostream>
#include "mldata.h"

namespace puml {

enum class dt_node_type : ml_uint {
  split = 0,
  leaf
}; 


enum class dt_comparison_op : ml_uint {
  noop = 0,
  lessthanorequal,
  greaterthan,
  equal,
  notequal,
};


struct dt_node;
using dt_node_ptr = std::shared_ptr<dt_node>;

struct dt_node {
  dt_node_type node_type;
  ml_uint feature_index;
  ml_feature_type feature_type;
  ml_feature_value feature_value = {};

  dt_comparison_op split_left_op;    
  dt_node_ptr split_left_node = nullptr;

  dt_comparison_op split_right_op;    
  dt_node_ptr split_right_node = nullptr;
  
  ml_data leaf_instances;
};


struct dt_feature_importance {
  ml_double sum_score_delta;
  ml_uint count;
};

struct dt_split;

class decision_tree final {

 public:

  decision_tree() {}

  decision_tree(const ml_string &path) { restore(path); }

  decision_tree(const ml_string &path, 
		const ml_instance_definition &mlid) { restore(path, mlid); }

  decision_tree(const ml_instance_definition &mlid, 
		const ml_string &feature_to_predict, 
		ml_uint max_tree_depth, ml_uint min_leaf_instances, 
		ml_uint features_to_consider_per_node=0, ml_uint seed=ML_DEFAULT_SEED,
		bool keep_instances_at_leaf_nodes=false);

  decision_tree(const ml_instance_definition &mlid, 
		ml_uint index_of_feature_to_predict, 
		ml_uint max_tree_depth, ml_uint min_leaf_instances, 
		ml_uint features_to_consider_per_node=0, ml_uint seed=ML_DEFAULT_SEED,
		bool keep_instances_at_leaf_nodes=false);

  //
  // A tree within an ensemble (random_forest, boosted_trees) writes
  // just its tree json without the mlid.  A single tree gets its own
  // directory with save tree and mlid json.
  //
  bool save(const ml_string &path, bool part_of_ensemble=false) const;

  bool restore(const ml_string &path);
  bool restore(const ml_string &path, const ml_instance_definition &mlid);

  //
  // Build the tree from data using parameters (max_tree_depth, etc)
  //
  bool train(const ml_data &mld);

  //
  // Evaluate the tree for the given instance and return the prediction as a ml_feature_value.
  // Use the continuous_value of the returned ml_feature_value if this is a regression tree,
  // and discrete_value_index for classification trees.
  //
  // Note: discrete_value_index is the internal mapping of the categorical feature value.  
  // Use the ml_instance_definition to map discrete_value_index to the actual categorical 
  // name (use tree.index_of_feature_to_predict() to get the index of the ml_feature_desc
  // in the ml_instance_definition. The ml_feature_desc has a vector of category names, 
  // discrete_values. Use discrete_values[discrete_value_index]).
  //
  ml_feature_value evaluate(const ml_instance &instance) const;

  // 
  // Summary includes tree type, structure, etc
  //
  ml_string summary() const;
  
  const ml_instance_definition &mlid() const { return(mlid_); }
  ml_uint index_of_feature_to_predict() const { return(index_of_feature_to_predict_); }
  ml_model_type type() const { return(type_); }
  ml_uint features_to_consider_per_node() const { return(features_to_consider_per_node_); }
  const ml_vector<dt_feature_importance> &feature_importance() const { return(feature_importance_); }
  const ml_string &name() const { return(name_); }
  ml_uint seed() const { return(seed_); }
  dt_node_ptr &root() { return(root_); }

  void set_name(const ml_string &name) { name_ = name; }
  void set_seed(ml_uint seed) { seed_ = seed; rng_ = ml_rng{seed}; }
  void set_max_tree_depth(ml_uint depth) { max_tree_depth_ = depth; }
  

 private:

  // tree build parameters
  ml_instance_definition mlid_;
  ml_uint index_of_feature_to_predict_ = 0;
  ml_uint max_tree_depth_ = 0;
  ml_uint min_leaf_instances_ = 0;
  ml_uint features_to_consider_per_node_ = 0; 
  ml_uint seed_ = ML_DEFAULT_SEED;
  bool keep_instances_at_leaf_nodes_ = false;

  // tree structure
  ml_model_type type_;
  ml_uint nodes_ = 0;
  ml_uint leaves_ = 0;
  dt_node_ptr root_ = nullptr;

  // misc
  ml_string name_;
  ml_rng rng_;
  ml_vector<dt_feature_importance> feature_importance_;

  // implementation 
  bool validate_for_training(const ml_data &mld);
  void build_tree_node(const ml_data &mld, dt_node_ptr &node, ml_uint depth, ml_double score);
  void config_leaf_node(const ml_data &mld, dt_node_ptr &leaf);
  bool prune_twin_leaf_nodes(dt_node_ptr &node);
  bool find_best_split(const ml_data &mld, dt_split &best_split, ml_double score);
  bool create_decision_tree_from_json(cJSON *json_object);
};


} // namespace puml

