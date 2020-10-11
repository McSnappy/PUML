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
#include <functional>

namespace puml {

using boosted_progress_callback = std::function<bool (ml_uint iteration)>;
using boosted_loss_func = std::function<ml_double (ml_double yi, ml_double yhat)>;
using boosted_gradient_func = std::function<ml_double (ml_double yi, ml_double yhat)>;


class boosted_trees final {

 public:

  static const ml_uint BT_DEFAULT_DEPTH = 4;
  static constexpr ml_float BT_DEFAULT_SUBSAMPLE_HALF = 0.5;
  static const ml_uint BT_DEFAULT_FEATURES_HALF = 0; 
  static const ml_uint BT_DEFAULT_MININST = 2;

  boosted_trees(const ml_string &path) { restore(path); }

  boosted_trees(const ml_instance_definition &mlid,
		const ml_string &feature_to_predict,
		ml_uint number_of_trees,
		ml_float learning_rate,
		ml_uint seed = ML_DEFAULT_SEED,
		ml_uint max_tree_depth = BT_DEFAULT_DEPTH,
		ml_float subsample = BT_DEFAULT_SUBSAMPLE_HALF,
		ml_uint min_leaf_instances = BT_DEFAULT_MININST,
		ml_uint features_to_consider = BT_DEFAULT_FEATURES_HALF);

  bool save(const ml_string &path) const;
  bool restore(const ml_string &path);

  bool train(const ml_data &mld);
  ml_feature_value evaluate(const ml_instance &instance) const;

  ml_string summary() const;
  
  const ml_instance_definition &mlid() const { return(mlid_); }
  ml_uint index_of_feature_to_predict() const { return(index_of_feature_to_predict_); }
  ml_model_type type() const { return(type_); }

  void set_progress_callback(boosted_progress_callback callback) { progress_callback_ = callback; }
  void set_loss_func(boosted_loss_func loss_func) { loss_func_ = loss_func; }
  void set_gradient_func(boosted_gradient_func grad_func) { gradient_func_ = grad_func; }
  

 private:
  
  // build parameters
  ml_instance_definition mlid_;
  ml_uint index_of_feature_to_predict_ = 0;
  ml_uint number_of_trees_ = 0;
  ml_float learning_rate_ = 0;
  ml_uint seed_ = ML_DEFAULT_SEED;  
  ml_uint max_tree_depth_ = BT_DEFAULT_DEPTH; 
  ml_float subsample_ = BT_DEFAULT_SUBSAMPLE_HALF;
  ml_uint min_leaf_instances_ = BT_DEFAULT_MININST;
  ml_uint features_to_consider_per_node_ = BT_DEFAULT_FEATURES_HALF;

  // loss defaults to squared error unless overridden
  boosted_loss_func loss_func_ = nullptr; 
  boosted_gradient_func gradient_func_ = nullptr;

  // ensemble structure
  ml_model_type type_;
  ml_vector<decision_tree> trees_;

  // optional progress callback excercised after each iteration.
  // return false to stop training
  boosted_progress_callback progress_callback_ = nullptr;

  // implementation
  bool read_boosted_trees_base_info_from_file(const ml_string &path);
  bool write_boosted_trees_base_info_to_file(const ml_string &path) const;
};


} // namespace puml





