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

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <limits>
#include <sstream>
#include <chrono>

#include "decisiontree.h"


typedef struct {
  ml_uint split_feature_index;
  ml_feature_type split_feature_type;
  ml_feature_value split_feature_value;
  dt_comparison_operator split_left_op;
  dt_comparison_operator split_right_op;
  ml_double left_score;
  ml_double right_score;
} dt_split;


#define DT_COMPARISON_EQUAL_TOL 0.00000001


bool _dt_validateInput(const ml_instance_definition &mlid, const ml_data &mld, const dt_build_config &dtbc) {

  if(mlid.size() == 0) {
    ml_log_error("empty instance definition...\n");
    return(false);
  }

  if(mld.size() == 0) {
    ml_log_error("empty instance data set...\n");
    return(false);
  }

  if(mlid.size() != (*mld[0]).size()) {
    ml_log_error("feature count mismatch b/t instance definition and instance data\n");
    return(false);
  }

  if(dtbc.index_of_feature_to_predict >= mlid.size()) {
    ml_log_error("invalid index of feature to predict...\n");
    return(false);
  }

  if(dtbc.min_leaf_instances == 0) {
    ml_log_error("minimum leaf instances must be greater than 0\n");
    return(false);
  }

  if(((dtbc.max_continuous_feature_splits > 0) || (dtbc.features_to_consider_per_node > 0)) && !dtbc.rng_config) {
    ml_log_error("missing rng_config...\n");
    return(false);
  }

  return(true);  
}

ml_double _dt_calcMeanForContinuousFeature(ml_uint feature_index, const ml_data &mld) {
  ml_double sum = 0.0;
  ml_uint count = 0;

  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    sum += (*mld[ii])[feature_index].continuous_value;
    ++count;
  }

  ml_double mean = (count > 0) ? (sum / count) : 0.0;
  return(mean);
}

ml_uint _dt_calcModeValueIndexForDiscreteFeature(ml_uint feature_index, const ml_data &mld) {

  ml_map<ml_uint, ml_uint> value_map;

  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    ml_uint discrete_value_index = (*mld[ii])[feature_index].discrete_value_index;
    value_map[discrete_value_index] += 1;
  }

  ml_uint mindex = 0, mmax = 0;
  for(ml_map<ml_uint, ml_uint>::iterator it = value_map.begin(); it != value_map.end(); ++it) {
    if(it->second > mmax) {
      mindex = it->first;
      mmax = it->second;
    }
  }

  return(mindex);
}

bool _dt_continuousFeatureSatisfiesConstraint(const ml_feature_value &feature_value, const ml_feature_value &split_feature_value, dt_comparison_operator op) {
  switch(op) {
  case DT_COMPARISON_OP_LESSTHANOREQUAL: return((feature_value.continuous_value < split_feature_value.continuous_value)); break; // OREQUAL, ha...
  case DT_COMPARISON_OP_GREATERTHAN: return((feature_value.continuous_value > split_feature_value.continuous_value)); break; 
  default: ml_log_error("confused by invalid split comparison operator %d (cont)... exiting.\n", op); exit(1); break;
  }

  return(false);
}

bool _dt_discreteFeatureSatisfiesConstraint(const ml_feature_value &feature_value, const ml_feature_value &split_feature_value, dt_comparison_operator op) {
  switch(op) {
  case DT_COMPARISON_OP_EQUAL: return((feature_value.discrete_value_index == split_feature_value.discrete_value_index)); break;
  case DT_COMPARISON_OP_NOTEQUAL: return((feature_value.discrete_value_index != split_feature_value.discrete_value_index)); break;
  default: ml_log_error("confused by invalid split comparison operator %d (disc)... exiting.\n", op); exit(1); break;
  }

  return(false);
}

bool _dt_instanceSatisfiesLeftConstraintOfSplit(const ml_instance &instance, ml_uint split_feature_index, ml_feature_type split_feature_type, 
						const ml_feature_value &split_feature_value, dt_comparison_operator split_left_op) {
 
  if(split_feature_index >= instance.size()) {
    ml_log_error("invalid split feature index: %d. exiting\n", split_feature_index); 
    exit(1);
  }
  
  const ml_feature_value &mlfv = instance[split_feature_index];

  switch(split_feature_type) {
  case ML_FEATURE_TYPE_CONTINUOUS: return(_dt_continuousFeatureSatisfiesConstraint(mlfv, split_feature_value, split_left_op)); break;
  case ML_FEATURE_TYPE_DISCRETE: return(_dt_discreteFeatureSatisfiesConstraint(mlfv, split_feature_value, split_left_op)); break;
  default: ml_log_error("confused by split feature type... exiting.\n"); exit(1); break;
  }
    
  return(false);
}

void _dt_performSplit(const ml_data &mld, const dt_split &split, ml_data &leftmld, ml_data &rightmld) {
 
  leftmld.reserve(mld.size());
  rightmld.reserve(mld.size());

  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    if(_dt_instanceSatisfiesLeftConstraintOfSplit(*mld[ii], split.split_feature_index, split.split_feature_type, split.split_feature_value, split.split_left_op)) {
      leftmld.push_back(mld[ii]);
    }
    else {
      rightmld.push_back(mld[ii]);
    }
  }
}

void _dt_addSplitsForDiscreteFeature(const ml_feature_desc &mlfd, ml_uint feature_index, ml_vector<dt_split> &splits) {
  
  // if this is a binary feature (plus missing category) we only need to consider one class 
  std::size_t end_index = (mlfd.discrete_values.size() == 3) ? 2 : mlfd.discrete_values.size(); 
  
  for(std::size_t ii=(mlfd.preserve_missing ? 0 : 1); ii < end_index; ++ii) { // 0 index is the <unknown> category
    dt_split dsplit;
    dsplit.split_feature_index = feature_index;
    dsplit.split_feature_type = ML_FEATURE_TYPE_DISCRETE;
    dsplit.split_feature_value.discrete_value_index = ii;
    dsplit.split_right_op = DT_COMPARISON_OP_EQUAL;
    dsplit.split_left_op = DT_COMPARISON_OP_NOTEQUAL;
    splits.push_back(dsplit);
  }
  
}

void _dt_addSplitsForContinuousFeature(const ml_feature_desc &mlfd, ml_uint feature_index, const ml_data &mld, const dt_build_config &dtbc, ml_vector<dt_split> &splits) {

  ml_vector<dt_split> feature_splits;
  ml_vector<ml_float> column;
  column.reserve(mld.size());

  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    column.push_back((*mld[ii])[feature_index].continuous_value);
  }

  std::sort(column.begin(), column.end());
  
  for(std::size_t ii=0; ii < (column.size() - 1); ++ii) {

    if(fabs(column[ii] - column[ii+1]) < DT_COMPARISON_EQUAL_TOL) {
      continue;
    }

    dt_split csplit;
    csplit.split_feature_index = feature_index;
    csplit.split_feature_type = ML_FEATURE_TYPE_CONTINUOUS;
    csplit.split_feature_value.continuous_value = (column[ii] + column[ii+1]) / 2.0;
    csplit.split_right_op = DT_COMPARISON_OP_GREATERTHAN;
    csplit.split_left_op = DT_COMPARISON_OP_LESSTHANOREQUAL;
    feature_splits.push_back(csplit);    

  }

  if((dtbc.max_continuous_feature_splits > 0) && dtbc.rng_config && (feature_splits.size() > dtbc.max_continuous_feature_splits)) {
    ml_shuffleVector(feature_splits, dtbc.rng_config);
    feature_splits.resize(dtbc.max_continuous_feature_splits);
  }

  if(feature_splits.size() > 0) {
    splits.insert(splits.end(), feature_splits.begin(), feature_splits.end());
  }

}

ml_double _dt_scoreRegion(const ml_data &mld, const dt_tree &tree) {

  if(mld.size() == 0) {
    return(0.0);
  }
  
  ml_double score = 0.0;

  if(tree.type == DT_TREE_TYPE_REGRESSION) {
    //MSE
    ml_double mean = _dt_calcMeanForContinuousFeature(tree.index_of_feature_to_predict, mld);
    for(std::size_t ii=0; ii < mld.size(); ++ii) {
      ml_double diff = (*mld[ii])[tree.index_of_feature_to_predict].continuous_value - mean;
      score += (diff * diff);
    }

  }
  else {
    //Gini Index
    ml_map<ml_uint, ml_uint> value_map;
    for(std::size_t ii=0; ii < mld.size(); ++ii) {
      ml_uint discrete_value_index = (*mld[ii])[tree.index_of_feature_to_predict].discrete_value_index;
      value_map[discrete_value_index] += 1;
    }

    for(ml_map<ml_uint, ml_uint>::iterator it = value_map.begin(); it != value_map.end(); ++it) {
      ml_double class_proportion = ((ml_double) it->second / mld.size());
      score += (class_proportion * (1.0 - class_proportion));
    }
  }

  return(score);
}

void _dt_pickRandomFeaturesToConsider(const ml_instance_definition &mlid, const dt_build_config &dtbc, ml_map<ml_uint, bool> &random_features) {

  if(!dtbc.rng_config || (dtbc.features_to_consider_per_node > (mlid.size() - 1))) {
    ml_log_warn("invalid random features config... considering all features.\n");
    return;
  }

  //ml_log("considering random features of the set: ");

  while(random_features.size() < dtbc.features_to_consider_per_node) {
    
    ml_uint feature_index = ml_generateRandomNumber(dtbc.rng_config) % mlid.size();

    if(feature_index == dtbc.index_of_feature_to_predict) {
      continue;
    }

    if(random_features.find(feature_index) != random_features.end()) {
      continue;
    }

    //ml_log("%u ", feature_index);

    random_features[feature_index] = true;
  }

  //ml_log("\n");
}

bool _dt_findBestSplit(const ml_instance_definition &mlid, const ml_data &mld, const dt_build_config &dtbc, dt_tree &tree, dt_split &best_split, ml_double score) {

  ml_vector<dt_split> splits;
  
  ml_map<ml_uint, bool> random_features_to_consider;
  if(dtbc.features_to_consider_per_node > 0) {
    _dt_pickRandomFeaturesToConsider(mlid, dtbc, random_features_to_consider);
  }

  for(std::size_t findex = 0; findex < mlid.size(); ++findex) {

    if(findex == tree.index_of_feature_to_predict) {
      continue;
    }

    if((random_features_to_consider.size() > 0) && (random_features_to_consider.find(findex) == random_features_to_consider.end())) {
      continue;
    }

    switch(mlid[findex].type) {
    case ML_FEATURE_TYPE_DISCRETE: _dt_addSplitsForDiscreteFeature(mlid[findex], findex, splits); break;
    case ML_FEATURE_TYPE_CONTINUOUS: _dt_addSplitsForContinuousFeature(mlid[findex], findex, mld, dtbc, splits); break;
    default: ml_log_error("invalid feature type...\n"); break;
    }
  }

  //ml_log("total splits to consider: %zu\n", splits.size());

  ml_double best_score,best_left_score,best_right_score;
  ml_uint best_split_index = 0;

  best_score = best_left_score = best_right_score = std::numeric_limits<ml_double>::max();

  for(std::size_t ii = 0; ii < splits.size(); ++ii) {

    ml_data leftmld, rightmld;
    _dt_performSplit(mld, splits[ii], leftmld, rightmld);
    
    ml_double lscore = _dt_scoreRegion(leftmld, tree);
    ml_double rscore = _dt_scoreRegion(rightmld, tree);
    ml_double combined_score = lscore + rscore;
    if(tree.type == DT_TREE_TYPE_CLASSIFICATION) {
      combined_score = (((ml_double) leftmld.size() / mld.size()) * lscore) + (((ml_double) rightmld.size() / mld.size()) * rscore);
    }

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

    tree.feature_importance[best_split.split_feature_index].sum_score_delta += (score - best_score);
    tree.feature_importance[best_split.split_feature_index].count += 1;

    return(true);
  }

  best_split.split_feature_index = 0;
  best_split.split_left_op = best_split.split_right_op = DT_COMPARISON_OP_NOOP;
  return(false);
}

void _dt_configLeafNode(const ml_instance_definition &mlid, const ml_data &mld, const dt_build_config &dtbc, dt_tree &tree, dt_node *leaf) {
  tree.leaves += 1;
  leaf->node_type = DT_NODE_TYPE_LEAF;
  leaf->feature_index = tree.index_of_feature_to_predict;
  leaf->feature_type = mlid[tree.index_of_feature_to_predict].type;
  if(tree.type == DT_TREE_TYPE_REGRESSION) {
    leaf->feature_value.continuous_value = _dt_calcMeanForContinuousFeature(leaf->feature_index, mld);
  }
  else {
    leaf->feature_value.discrete_value_index = _dt_calcModeValueIndexForDiscreteFeature(leaf->feature_index, mld);
  }
}

void _dt_configSplitNode(const dt_split &split, dt_node *split_node) {
  split_node->node_type = DT_NODE_TYPE_SPLIT;
  split_node->feature_index = split.split_feature_index;
  split_node->feature_type = split.split_feature_type;
  split_node->feature_value = split.split_feature_value;
  split_node->split_left_op = split.split_left_op;
  split_node->split_right_op = split.split_right_op;
}

bool _dt_pruneTwinLeafNodes(dt_node *node, dt_tree &tree) {

  // 
  // we prune sibling leaf nodes that predict the same class/value and
  // convert their parent from split to leaf node.
  //
 
  if((node->split_left_node->node_type == DT_NODE_TYPE_LEAF) && (node->split_right_node->node_type == DT_NODE_TYPE_LEAF)) {

    if(((tree.type == DT_TREE_TYPE_CLASSIFICATION) && (node->split_left_node->feature_value.discrete_value_index == node->split_right_node->feature_value.discrete_value_index)) ||
       ((tree.type == DT_TREE_TYPE_REGRESSION) && (fabs(node->split_left_node->feature_value.continuous_value - node->split_right_node->feature_value.continuous_value) < DT_COMPARISON_EQUAL_TOL))) {
      tree.nodes -= 2;
      tree.leaves -= 2;
      delete node->split_left_node;
      delete node->split_right_node;
      node->split_left_node = nullptr;
      node->split_right_node = nullptr;
      return(true);
    }
  }

  return(false);
}

void _dt_buildTreeNode(const ml_instance_definition &mlid, const ml_data &mld, const dt_build_config &dtbc, dt_tree &tree, dt_node **node, ml_uint depth, ml_double score) {

  *node = new dt_node;
  if(!*node) {
    ml_log_error("out of memory. aborting...\n");
    abort();
  }

  (*node)->split_left_node = nullptr;
  (*node)->split_right_node = nullptr;

  tree.nodes += 1;

  if((dtbc.max_tree_depth > 0) && (depth == dtbc.max_tree_depth)) {
    _dt_configLeafNode(mlid, mld, dtbc, tree, *node);
    return;
  }
  
  dt_split best_split;
  ml_data leftmld, rightmld;

  if(_dt_findBestSplit(mlid, mld, dtbc, tree, best_split, score)) {
    _dt_performSplit(mld, best_split, leftmld, rightmld);
  }

  if((leftmld.size() < dtbc.min_leaf_instances) || 
     (rightmld.size() < dtbc.min_leaf_instances)) {
    _dt_configLeafNode(mlid, mld, dtbc, tree, *node);
    return;
  }

  _dt_configSplitNode(best_split, *node);

  _dt_buildTreeNode(mlid, leftmld, dtbc, tree, &((*node)->split_left_node), depth+1, best_split.left_score);
  _dt_buildTreeNode(mlid, rightmld, dtbc, tree, &((*node)->split_right_node), depth+1, best_split.right_score);

 
  if(_dt_pruneTwinLeafNodes(*node, tree)) {
    _dt_configLeafNode(mlid, mld, dtbc, tree, *node);
  }
 

}

bool dt_buildDecisionTree(const ml_instance_definition &mlid, const ml_data &mld, const dt_build_config &dtbc, dt_tree &tree) {

  if(!_dt_validateInput(mlid, mld, dtbc)) {
    return(false);
  }

  tree.root = nullptr;
  tree.nodes = tree.leaves = 0;
  tree.feature_importance.clear();
  tree.feature_importance.resize(mlid.size());
  tree.index_of_feature_to_predict = dtbc.index_of_feature_to_predict;
  tree.type = (mlid[dtbc.index_of_feature_to_predict].type == ML_FEATURE_TYPE_DISCRETE) ? DT_TREE_TYPE_CLASSIFICATION : DT_TREE_TYPE_REGRESSION;

  auto t1 = std::chrono::high_resolution_clock::now();
  _dt_buildTreeNode(mlid, mld, dtbc, tree, &tree.root, 0, _dt_scoreRegion(mld, tree)); 
  auto t2 = std::chrono::high_resolution_clock::now();
   
  ml_uint ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  ml_log("built tree %s in %.3f seconds (%u leaves, %u nodes)\n", tree.name.c_str(), (ms / 1000.0), tree.leaves, tree.nodes); 

  return(true);
}

ml_string _dt_nameForSplitOperator(dt_comparison_operator op) {
  
  ml_string op_name;

  switch(op) {
  case DT_COMPARISON_OP_NOOP : op_name = "no-op"; break;
  case DT_COMPARISON_OP_LESSTHANOREQUAL: op_name = "<="; break;
  case DT_COMPARISON_OP_GREATERTHAN: op_name = ">"; break;
  case DT_COMPARISON_OP_EQUAL: op_name = "="; break;
  case DT_COMPARISON_OP_NOTEQUAL: op_name = "!="; break;
  default: ml_log_error("invalid split operator"); break;
  }

  return(op_name);
}

void _dt_printDecisionTreeNode(const ml_instance_definition &mlid, const dt_node *node, ml_uint depth) {

  ml_string feature_value_as_string;
  if(node->feature_type == ML_FEATURE_TYPE_DISCRETE) {
    feature_value_as_string = mlid[node->feature_index].discrete_values[node->feature_value.discrete_value_index];
  }
  else {
    std::ostringstream ss;
    ss << node->feature_value.continuous_value;
    feature_value_as_string = ss.str(); 
  }

  if(node->node_type == DT_NODE_TYPE_SPLIT) {
    
    // Left Side
    ml_log("\n");
    for(ml_uint ii=0; ii < depth; ++ii) {
      ml_log("|  ");
    }

    ml_log("%s %s %s", mlid[node->feature_index].name.c_str(), _dt_nameForSplitOperator(node->split_left_op).c_str(), feature_value_as_string.c_str());
    _dt_printDecisionTreeNode(mlid, node->split_left_node, depth+1);
  
    // Right Side
    ml_log("\n");
    for(ml_uint ii=0; ii < depth; ++ii) {
      ml_log("|  ");
    }

    ml_log("%s %s %s", mlid[node->feature_index].name.c_str(), _dt_nameForSplitOperator(node->split_right_op).c_str(), feature_value_as_string.c_str());
    _dt_printDecisionTreeNode(mlid, node->split_right_node, depth+1);
  }
  else {
    // DT_NODE_TYPE_LEAF
    ml_log(": %s", feature_value_as_string.c_str());
  }
  
}

void dt_printDecisionTreeSummary(const ml_instance_definition &mlid, const dt_tree &tree) {
  
  ml_log("\n\n*** Decision Tree Summary ***\n\n");
  ml_log("Feature To Predict: %s\n", mlid[tree.index_of_feature_to_predict].name.c_str());
  ml_log("Type: %s, Leaves: %d, Size: %d\n", (tree.type == DT_TREE_TYPE_REGRESSION) ? "regression" : "classification", tree.leaves, tree.nodes);
  _dt_printDecisionTreeNode(mlid, tree.root, 0);
  ml_log("\n\n");
}


const ml_feature_value *_dt_evaluateDecisionTreeNodeForInstance(const dt_node *node, const ml_instance &instance) {

  if(node->node_type == DT_NODE_TYPE_LEAF) {
    return(&node->feature_value);
  }

  if(_dt_instanceSatisfiesLeftConstraintOfSplit(instance, node->feature_index, node->feature_type, node->feature_value, node->split_left_op)) {
    return(_dt_evaluateDecisionTreeNodeForInstance(node->split_left_node, instance));
  }

  return(_dt_evaluateDecisionTreeNodeForInstance(node->split_right_node, instance));
}

const ml_feature_value *dt_evaluateDecisionTreeForInstance(const ml_instance_definition &mlid, const dt_tree &tree, const ml_instance &instance) {
  
  if(instance.size() != mlid.size()) {
    ml_log_error("feature count mismatch b/t instance definition and instance to evaluate\n");
    return(nullptr);
  }

  return(_dt_evaluateDecisionTreeNodeForInstance(tree.root, instance));
}

void dt_printDecisionTreeResultsForData(const ml_instance_definition &mlid, const ml_data &mld, const dt_tree &tree) {

  ml_regression_results mlrr = {};
  ml_classification_results mlcr = {};
  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    const ml_feature_value *result = dt_evaluateDecisionTreeForInstance(mlid, tree, *mld[ii]);
    switch(tree.type) {
    case DT_TREE_TYPE_CLASSIFICATION: ml_collectClassificationResultForInstance(mlid, tree.index_of_feature_to_predict, *mld[ii], result, mlcr); break;
    case DT_TREE_TYPE_REGRESSION: ml_collectRegressionResultForInstance(mlid, tree.index_of_feature_to_predict, *mld[ii], result, mlrr); break;
    default: break;
    }
  }
  
  switch(tree.type) {
  case DT_TREE_TYPE_CLASSIFICATION: ml_printClassificationResultsSummary(mlid, tree.index_of_feature_to_predict, mlcr); break;
  case DT_TREE_TYPE_REGRESSION: ml_printRegressionResultsSummary(mlrr); break;
  default: break;
  }

}

cJSON *_dt_addNodesToJSONObject(const dt_node *node, cJSON *json_nodes, ml_uint &node_id) {

  if(!node) {
    return(nullptr);
  }

  cJSON *json_node = cJSON_CreateObject();
  cJSON_AddNumberToObject(json_node, "id", node_id);
  cJSON_AddNumberToObject(json_node, "nt", node->node_type);
  cJSON_AddNumberToObject(json_node, "fi", node->feature_index);
  cJSON_AddNumberToObject(json_node, "ft", node->feature_type);
  cJSON_AddNumberToObject(json_node, "fv", 
			  (node->feature_type == ML_FEATURE_TYPE_CONTINUOUS) ? node->feature_value.continuous_value : node->feature_value.discrete_value_index);

  //
  // we don't use cJSON_AddItemToArray(json_nodes, json_node) here for performance reasons
  //
  if(node_id == 0) { // root node
    json_nodes->child = json_node;
  }
  else {
    json_nodes->next = json_node;
    json_node->prev = json_nodes;
  }
  
  json_nodes = json_node;

  if(node->split_left_node) {
    ml_uint left_node_id = ++node_id;
    cJSON_AddNumberToObject(json_node, "lid", left_node_id);
    cJSON_AddNumberToObject(json_node, "lop", node->split_left_op);
    json_nodes = _dt_addNodesToJSONObject(node->split_left_node, json_nodes, node_id); 
  }

  if(node->split_right_node) {
    ml_uint right_node_id = ++node_id;
    cJSON_AddNumberToObject(json_node, "rid", right_node_id);
    cJSON_AddNumberToObject(json_node, "rop", node->split_right_op);
    json_nodes = _dt_addNodesToJSONObject(node->split_right_node, json_nodes, node_id);
  }

  return(json_nodes);
}

cJSON *_dt_createJSONObjectFromDecisionTree(const dt_tree &tree) {
  
  cJSON *json_tree = cJSON_CreateObject();
  cJSON_AddStringToObject(json_tree, "object", "dt_tree");
  cJSON_AddNumberToObject(json_tree, "type", tree.type);
  cJSON_AddNumberToObject(json_tree, "index_of_feature_to_predict", tree.index_of_feature_to_predict);

  cJSON *json_nodes = cJSON_CreateArray();
  cJSON_AddItemToObject(json_tree, "nodes", json_nodes);

  ml_uint node_id = 0;
  _dt_addNodesToJSONObject(tree.root, json_nodes, node_id);

  return(json_tree);
}

bool _dt_validateTreeNodeJSONObject(cJSON *json_node) {
  if(!cJSON_GetObjectItem(json_node, "nt") ||
     !cJSON_GetObjectItem(json_node, "fi") ||
     !cJSON_GetObjectItem(json_node, "ft") ||
     !cJSON_GetObjectItem(json_node, "fv")) {
    return(false);
  }
  
  dt_node_type node_type = (dt_node_type) cJSON_GetObjectItem(json_node, "nt")->valueint;
  if(node_type == DT_NODE_TYPE_SPLIT) {
    if(!cJSON_GetObjectItem(json_node, "lop") ||
       !cJSON_GetObjectItem(json_node, "lid") ||
       !cJSON_GetObjectItem(json_node, "rop") ||
       !cJSON_GetObjectItem(json_node, "rid")) {
      return(false);
    }
  }

  return(true);
}

bool _dt_createTreeNodeFromJSONObject(dt_tree &tree, dt_node **node, ml_uint node_id, ml_map<ml_uint, cJSON *> &nodes_map) {

  cJSON *json_node = (nodes_map.find(node_id) != nodes_map.end()) ? nodes_map[node_id] : nullptr;
  if(!json_node) {
    ml_log_error("can't find node in json with node_id of %d\n", node_id);
    return(false);
  }

  if(!_dt_validateTreeNodeJSONObject(json_node)) {
    ml_log_error("invalid or incomplete node json. node id: %u\n", node_id);
    return(false);
  }

  *node = new dt_node;
  if(!*node) {
    ml_log_error("out of memory...\n");
    return(false);
  }

  tree.nodes += 1;

  (*node)->node_type = (dt_node_type) cJSON_GetObjectItem(json_node, "nt")->valueint;
  (*node)->feature_index = cJSON_GetObjectItem(json_node, "fi")->valueint;
  (*node)->feature_type = (ml_feature_type) cJSON_GetObjectItem(json_node, "ft")->valueint;
  if((*node)->feature_type == ML_FEATURE_TYPE_CONTINUOUS) {
    (*node)->feature_value.continuous_value = cJSON_GetObjectItem(json_node, "fv")->valuedouble;
  }
  else {
    (*node)->feature_value.discrete_value_index = cJSON_GetObjectItem(json_node, "fv")->valueint;
  }

  if((*node)->node_type == DT_NODE_TYPE_LEAF) {
    tree.leaves += 1;
  }
  else {
    (*node)->split_left_op = (dt_comparison_operator) cJSON_GetObjectItem(json_node, "lop")->valueint;
    ml_uint left_node_id = cJSON_GetObjectItem(json_node, "lid")->valueint;

    (*node)->split_right_op = (dt_comparison_operator) cJSON_GetObjectItem(json_node, "rop")->valueint;
    ml_uint right_node_id = cJSON_GetObjectItem(json_node, "rid")->valueint;

    if(!_dt_createTreeNodeFromJSONObject(tree, &((*node)->split_left_node), left_node_id, nodes_map) ||
       !_dt_createTreeNodeFromJSONObject(tree, &((*node)->split_right_node), right_node_id, nodes_map)) {
      return(false);
    }
  }
   

  return(true);
}

bool _dt_createDecisionTreeFromJSONObject(cJSON *json_object, dt_tree &tree) {

  if(!json_object) {
    ml_log_error("nil json object...\n");
    return(false);
  }

  cJSON *object = cJSON_GetObjectItem(json_object, "object");
  if(!object || !object->valuestring || strcmp(object->valuestring, "dt_tree")) {
    ml_log_error("json object is not a decision tree...\n");
    return(false);
  }

  cJSON *type = cJSON_GetObjectItem(json_object, "type");
  if(!type || (type->type != cJSON_Number)) {
    ml_log_error("json object is missing tree type\n");
    return(false);
  }

  tree.type = (dt_tree_type) type->valueint;

  cJSON *index = cJSON_GetObjectItem(json_object, "index_of_feature_to_predict");
  if(!index || (index->type != cJSON_Number)) {
    ml_log_error("json object is missing the index of the feature to predict\n");
    return(false);
  }

  tree.index_of_feature_to_predict = index->valueint;

  cJSON *nodes_array = cJSON_GetObjectItem(json_object, "nodes");
  if(!nodes_array || (nodes_array->type != cJSON_Array)) {
    ml_log_error("json object is missing a nodes array\n");
    return(false);
  }

  ml_map<ml_uint, cJSON *> nodes_map;
  cJSON *node = nodes_array->child;;
  while(node) {
    if(!node || (node->type != cJSON_Object)) {
      ml_log_error("json object has bogus node in nodes array\n");
      return(false);
    }
    
    cJSON *node_id = cJSON_GetObjectItem(node, "id");
    if(!node_id || (node_id->type != cJSON_Number)) {
      ml_log_error("json object has node with bogus node_id");
      return(false);
    }

    nodes_map[node_id->valueint] = node;
    node = node->next;
  }

  if(nodes_map.empty()) {
    ml_log_error("json object has empty nodes array\n");
    return(false);
  }

  if(!_dt_createTreeNodeFromJSONObject(tree, &tree.root, 0, nodes_map)) {
    ml_log_error("failed to build tree nodes from json...\n");
    return(false);
  }

  return(true);
}

bool dt_writeDecisionTreeToFile(const ml_string &path_to_file, const dt_tree &tree) {
  cJSON *json_object = _dt_createJSONObjectFromDecisionTree(tree);
  if(!json_object) {
    ml_log_error("couldn't create json object from tree\n");
    return(false);
  }

  bool status = ml_writeModelJSONToFile(path_to_file, json_object);
  cJSON_Delete(json_object);

  return(status);
}

bool dt_readDecisionTreeFromFile(const ml_string &path_to_file, dt_tree &tree) {
  cJSON *json_object = ml_readModelJSONFromFile(path_to_file);
  if(!json_object) {
    ml_log_error("couldn't load decision tree json object from model file: %s\n", path_to_file.c_str());
    return(false);
  }

  tree.root = nullptr;
  tree.leaves = tree.nodes = 0;
  bool status = _dt_createDecisionTreeFromJSONObject(json_object, tree);
  cJSON_Delete(json_object);

  return(status);
}

void _dt_freeDecisionTreeNode(dt_node *node) {
  if(!node) {
    return;
  }

  if(node->node_type == DT_NODE_TYPE_SPLIT) {
    _dt_freeDecisionTreeNode(node->split_left_node);
    _dt_freeDecisionTreeNode(node->split_right_node);
  }

  delete node;
}

void dt_freeDecisionTree(dt_tree &tree) {
  _dt_freeDecisionTreeNode(tree.root);
  tree.index_of_feature_to_predict = 0;
  tree.nodes = 0;
  tree.leaves = 0;
  tree.name = "";
  tree.root = nullptr;
}
