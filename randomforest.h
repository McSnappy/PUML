#ifndef __RANDOM_FOREST_H__
#define __RANDOM_FOREST_H__

#include "decisiontree.h"

typedef struct {

  ml_uint number_of_threads; 
  ml_uint number_of_trees;
  ml_uint index_of_feature_to_predict; // see ml_indexOfFeatureWithName()
  ml_uint max_tree_depth; // 0 for unlimited
  ml_uint max_continuous_feature_splits; // 0 to consider all splits [experimental, see decisiontree.h]
  ml_uint features_to_consider_per_node; // 0 to consider all features
  ml_uint seed;

} rf_build_config;


typedef struct {

  ml_uint index_of_feature_to_predict;
  dt_tree_type type;
  ml_vector<dt_tree> trees;

} rf_forest;


//
// Build the forest given instance definition, training data, and forest build config. 
// oob_for_mld is an optional parameter that will be populated with the out of bag
// prediction for each instance in the training data.
//
// returns true on success, false otherwise
//
bool rf_buildRandomForest(const ml_instance_definition &mlid, const ml_data &mld, const rf_build_config &rfbc, 
			  rf_forest &forest, ml_vector<ml_feature_value> *oob_for_mld = nullptr);


//
// Release memory and clear the forest structure
//
void rf_freeRandomForest(rf_forest &forest);


//
// Evaluates the instance over each tree in the forest and returns the average for regression, 
// or majority vote for classification. 
//
// Use the continuous_value of the predicted ml_feature_value if this is a regression tree,
// and use the discrete_value_index for classification trees. See dt_evaluateDecisionTreeForInstance()
// for details on how to map discrete_value_index to the actual category name.
//
// tree_predictions is an optional parameter that will be populated with the predicted ml_feature_value 
// for the given instance from each tree in the forest.
//
// returns true on success, otherwise false
//
bool rf_evaluateRandomForestForInstance(const ml_instance_definition &mlid, const rf_forest &forest, const ml_instance &instance, 
					ml_feature_value &prediction, ml_vector<ml_feature_value> *tree_predictions = nullptr);


//
// Evaluates every instance in mld and prints the regression or classification summary
//
void rf_printRandomForestResultsForData(const ml_instance_definition &mlid, const ml_data &mld, const rf_forest &forest);


//
// Read/Write Random Forest to disk, including the instance definition (JSON) 
//
bool rf_writeRandomForestToDirectory(const ml_string &path_to_dir, const ml_instance_definition &mlid, 
				     const rf_forest &forest, bool overwrite_existing = true);
bool rf_readRandomForestFromDirectory(const ml_string &path_to_dir, ml_instance_definition &mlid, rf_forest &forest);



#endif


