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

#ifndef __BOOSTING_H__
#define __BOOSTING_H__

#include "decisiontree.h"

namespace puml {

typedef struct {

  ml_float learning_rate;
  ml_uint number_of_trees;
  ml_uint index_of_feature_to_predict; // see ml_indexOfFeatureWithName()
  ml_uint features_to_consider_per_node; // 0 to consider all features
  ml_uint max_tree_depth; // 0 for unlimited
  ml_uint min_leaf_instances;
  ml_uint seed;
  ml_float subsample; // 0.5 to build each tree using 50% of the data randomly sampled

} boosted_build_config;


typedef struct {

  ml_uint index_of_feature_to_predict;
  ml_float learning_rate;
  dt_tree_type type;
  ml_vector<dt_tree> trees;

} boosted_trees;


//
// buildBoostedTrees will use the callback after each iteration. return false to stop the build process
//
typedef bool (*boostedBuildCallback)(const ml_instance_definition &mlid, const boosted_trees &bt, ml_uint iteration, void *user);


//
// NOTE: Implemented for regression only
//
// Build the boosted ensemble given instance definition, training data, and boosted build config. 
// Optional progress callback allows periodic validation scoring and early exit.
// returns true on success
//
bool buildBoostedTrees(const ml_instance_definition &mlid, const boosted_build_config &bbc,
		       const ml_mutable_data &mld, boosted_trees &bt, 
		       boostedBuildCallback callback = nullptr, void *user = nullptr);


//
// Release memory and clear the boosted trees structure
//
void freeBoostedTrees(boosted_trees &bt);


//
// Evaluates the instance over each tree in the ensemble scaled by the shrinkage parameter.
// Use the continuous_value of the predicted ml_feature_value 
//
// returns true on success, otherwise false
//
bool evaluateBoostedTreesForInstance(const ml_instance_definition &mlid, const boosted_trees &bt, 
				     const ml_instance &instance, ml_feature_value &prediction);


//
// Evaluates every instance in mld and prints the regression summary
//
void printBoostedTreesResultsForData(const ml_instance_definition &mlid, 
				     const ml_data &mld, const boosted_trees &bt);


//
// Read/Write Boosted Trees ensemble to disk, including the instance definition (JSON) 
//
bool writeBoostedTreesToDirectory(const ml_string &path_to_dir, const ml_instance_definition &mlid, 
				  const boosted_trees &bt, bool overwrite_existing = true);
bool readBoostedTreesFromDirectory(const ml_string &path_to_dir, ml_instance_definition &mlid, boosted_trees &bt);



} // namespace puml

#endif



