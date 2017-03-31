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
#include "machinelearning.h"

namespace puml {

typedef enum {

  DT_TREE_TYPE_CLASSIFICATION = 0,
  DT_TREE_TYPE_REGRESSION

} dt_tree_type;


typedef enum {

  DT_NODE_TYPE_SPLIT = 0,
  DT_NODE_TYPE_LEAF

} dt_node_type;


typedef enum {

  DT_COMPARISON_OP_NOOP = 0,
  DT_COMPARISON_OP_LESSTHANOREQUAL,
  DT_COMPARISON_OP_GREATERTHAN,
  DT_COMPARISON_OP_EQUAL,
  DT_COMPARISON_OP_NOTEQUAL

} dt_comparison_operator;


struct dt_node;
using dt_node_ptr = std::shared_ptr<dt_node>;

struct dt_node {

  dt_node_type node_type;
  ml_uint feature_index;
  ml_feature_type feature_type;
  ml_feature_value feature_value = {};

  dt_comparison_operator split_left_op;    
  dt_node_ptr split_left_node = nullptr;

  dt_comparison_operator split_right_op;    
  dt_node_ptr split_right_node = nullptr;
  
  ml_data leaf_instances;

};


typedef struct {

  ml_uint index_of_feature_to_predict; // see ml_indexOfFeatureWithName()
  ml_uint min_leaf_instances; 
  ml_uint max_tree_depth; 

  ml_uint max_continuous_feature_splits; // 0 to consider all splits. Otherwise,
                                         // take a random sample of possible splits
                                         // of the specified size [experimental]

  ml_uint features_to_consider_per_node; // 0 to consider all features
  ml_rng_config *rng_config; // used for random feature selection. can be nil

  bool keep_instances_at_leaf_nodes; // keep a copy of the instances at each leaf node 
                                     // (leaf_instances in dt_node, used during boosting)
} dt_build_config;


typedef struct {
  ml_double sum_score_delta;
  ml_uint count;
} dt_feature_importance;


typedef struct {

  ml_uint index_of_feature_to_predict;
  dt_tree_type type;
  ml_uint nodes = 0;
  ml_uint leaves = 0;
  dt_node_ptr root = nullptr;

  ml_string name;
  ml_vector<dt_feature_importance> feature_importance;

} dt_tree;


//
// Builds a decision tree given the instance definition, data, and tree build config.  
// returns true on success
//
bool buildDecisionTree(const ml_instance_definition &mlid, const ml_data &mld, const dt_build_config &dtbc, dt_tree &tree);


//
// Evaluates the tree given the instance and returns the prediction as a ml_feature_value.
// Use the continuous_value of the returned ml_feature_value if this is a regression tree,
// and use the discrete_value_index for classification trees.
//
// Note: discrete_value_index is the internal mapping of the categorical feature value.  Use the ml_instance_definition to map
// discrete_value_index to the actual categorical name (use tree.index_of_feature_to_predict to get the index of the ml_feature_desc
// from the ml_instance_definition. The ml_feature_desc has a vector of category names, discrete_values. Use discrete_values[discrete_value_index]).
//
// The tree owns the resources for the returned ml_feature_value
//
const ml_feature_value *evaluateDecisionTreeForInstance(const ml_instance_definition &mlid, const dt_tree &tree, const ml_instance &instance);


//
// Text based display of decision tree
//
void printDecisionTreeSummary(const ml_instance_definition &mlid, const dt_tree &tree);


//
// Evaluates every instance in mld and prints the regression or classification summary
//
void printDecisionTreeResultsForData(const ml_instance_definition &mlid, const ml_data &mld, const dt_tree &tree);


//
// Read/Write Decision Tree to disk (JSON)
//
bool writeDecisionTreeToFile(const ml_string &path_to_file, const dt_tree &tree);
bool readDecisionTreeFromFile(const ml_string &path_to_file, dt_tree &tree);


} // namespace puml





