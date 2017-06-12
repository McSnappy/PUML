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

#include "decisiontree.h"

namespace puml {

using rf_oob_indices = ml_set<ml_uint>;
using feature_importance_tuple = std::tuple<ml_uint, ml_string>; 


class random_forest final {

 public:

  static const ml_uint RF_DEFAULT_THREADS = 2;
  static const ml_uint RF_DEFAULT_DEPTH = 50;
  static const ml_uint RF_DEFAULT_MININST = 2;
  static const ml_uint RF_DEFAULT_FEATURES_SQRT = 0;

  random_forest(const ml_string &path) { restore(path); }

  random_forest(const ml_instance_definition &mlid,
		const ml_string &feature_to_predict,
		ml_uint number_of_trees,
		ml_uint seed = ML_DEFAULT_SEED,
		ml_uint number_of_threads = RF_DEFAULT_THREADS,
		ml_uint max_tree_depth = RF_DEFAULT_DEPTH, 
		ml_uint min_leaf_instances = RF_DEFAULT_MININST,
		ml_uint features_to_consider_per_node = RF_DEFAULT_FEATURES_SQRT);

  bool save(const ml_string &path) const;
  bool restore(const ml_string &path);
		
  bool train(const ml_data &mld);
  ml_feature_value evaluate(const ml_instance &instance) const;

  ml_string summary() const;
  ml_string feature_importance_summary() const;

  const ml_instance_definition &mlid() const { return(mlid_); }
  const ml_vector<decision_tree> &trees() const { return(trees_); }
  const ml_vector<ml_feature_value> &oob_predictions() const { return(oob_predictions_); }
  ml_uint index_of_feature_to_predict() const { return(index_of_feature_to_predict_); }
  ml_model_type type() const { return(type_); }

  void set_seed(ml_uint seed) { seed_ = seed;}
  void set_number_of_trees(ml_uint ntrees) { number_of_trees_ = ntrees; }
  void set_number_of_threads(ml_uint nthreads) { number_of_threads_ = nthreads; }
  void set_evaluate_oob(bool eval_oob) { evaluate_oob_ = eval_oob; }
  void set_trees(const ml_vector<decision_tree> &trees);

 private:

  // forest build parameters
  ml_instance_definition mlid_;
  ml_uint index_of_feature_to_predict_ = 0;
  ml_uint number_of_trees_ = 0;
  ml_uint seed_ = ML_DEFAULT_SEED;
  ml_uint number_of_threads_ = 0; 
  ml_uint max_tree_depth_ = 0; 
  ml_uint min_leaf_instances_ = 0;
  ml_uint features_to_consider_per_node_ = 0;
  bool evaluate_oob_ = false;

  // forest structure
  ml_model_type type_;
  ml_vector<decision_tree> trees_;

  // feature importance & out-of-bag error
  // (available after train(). neither are saved/restored)
  ml_vector<feature_importance_tuple> feature_importance_;
  ml_vector<ml_feature_value> oob_predictions_;

  // implementation
  bool single_threaded_train(const ml_data &mld, 
			     ml_vector<rf_oob_indices> &oobs, 
			     ml_vector<dt_feature_importance> &forest_feature_importance);

  bool multi_threaded_train(const ml_data &mld, 
			    ml_vector<rf_oob_indices> &oobs, 
			    ml_vector<dt_feature_importance> &forest_feature_importance);

  bool write_random_forest_base_info_to_file(const ml_string &path) const;
  bool read_random_forest_base_info_from_file(const ml_string &path);
  void evaluate_out_of_bag(const ml_data &mld, const ml_vector<rf_oob_indices> &oobs);
};


} // namespace puml





