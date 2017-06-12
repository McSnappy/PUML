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

#include "mldata.h"

namespace puml {

using knn_neighbor = std::pair<ml_double, ml_instance_ptr>; // distance and the instance 
  
class knn final {

 public:

  knn(const ml_instance_definition &mlid,
      const ml_string &feature_to_predict,
      const ml_uint k);

  bool save(const ml_string &path) const { return(false); }
  bool restore(const ml_string &path) { return(false); }
  
  bool train(const ml_data &mld);
  ml_feature_value evaluate(const ml_instance &instance) const;
  ml_feature_value evaluate(const ml_instance &instance, 
			    ml_vector<knn_neighbor> &neighbors) const;
  
  ml_string summary() const;
  
  const ml_instance_definition &mlid() const { return(mlid_); }
  ml_uint index_of_feature_to_predict() const { return(index_of_feature_to_predict_); }
  ml_model_type type() const { return(type_); }

  ml_uint k() const { return(k_); }
  void set_k(ml_uint k) { k_ = k; validated_ = false;}


 private:

  ml_instance_definition mlid_;
  ml_model_type type_;
  ml_uint index_of_feature_to_predict_ = 0;
  ml_uint k_ = 0;
  ml_data training_data_;
  bool validated_ = false;

};


} // namespace puml 

