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

#include "knn.h"
#include <algorithm>
#include <map>

namespace puml {


knn::knn(const ml_instance_definition &mlid,
	 const ml_string &feature_to_predict,
	 const ml_uint k) : 
  mlid_(mlid),
  index_of_feature_to_predict_(index_of_feature_with_name(feature_to_predict, mlid)),
  k_(k) {

  type_ = (mlid_[index_of_feature_to_predict_]->type == ml_feature_type::discrete) ? ml_model_type::classification : ml_model_type::regression;
}


static bool validate_input(const ml_instance_definition &mlid, const ml_data &mld, 
			   ml_uint k, ml_uint index_of_feature_to_predict) {
			
  if(mlid.empty()) {
    log_error("knn: empty instance definition...\n");
    return(false);
  }

  if(mld.empty()) {
    log_error("knn: empty training data...\n");
    return(false);
  }

  if(index_of_feature_to_predict >= mlid.size()) {
    log_error("knn: invalid feature to predict...\n");
    return(false);
  }

  if(k == 0) {
    log_error("knn: k must be > 0\n");
    return(false);
  }

  bool has_discrete_features = false;
  bool has_continuous_features = false;
  for(std::size_t ii=0; ii < mlid.size(); ++ii) {

    if(ii == index_of_feature_to_predict) {
      continue;
    }

    if(mlid[ii]->type == ml_feature_type::discrete) {
      has_discrete_features = true;
    }
    else {
      has_continuous_features = true;
    }
  }

  if(!has_continuous_features) {
    log_error("knn: no continuous features...");
    return(false);
  }

  if(has_discrete_features) {
    log_warn("knn: discrete (categorical) features will be ignored...\n");
  }

  return(true);
}


static void predict_using_neighbors(const ml_vector<knn_neighbor> &all_distances, 
				    const ml_instance_definition &mlid, 
				    ml_uint k, ml_uint index_of_feature_to_predict, 
				    ml_feature_value &prediction, 
				    ml_vector<knn_neighbor> &neighbors_considered) {

  ml_double continuous_sum = 0;
  ml_uint continuous_count = 0;
  std::map<ml_uint, ml_uint> discrete_mode_map;

  for(std::size_t dist_index = 0; (dist_index < k) && (dist_index < all_distances.size()); ++dist_index) {
    
    ml_instance_ptr neighbor = all_distances[dist_index].second;
    
    neighbors_considered.push_back(all_distances[dist_index]);

    if(mlid[index_of_feature_to_predict]->type == ml_feature_type::continuous) {
      continuous_sum += (*neighbor)[index_of_feature_to_predict].continuous_value;
      continuous_count += 1;
    }
    else {
      discrete_mode_map[(*neighbor)[index_of_feature_to_predict].discrete_value_index] += 1;
    }

  }

  if(mlid[index_of_feature_to_predict]->type == ml_feature_type::continuous) {
    prediction.continuous_value = (continuous_sum / continuous_count);
  }
  else {
    ml_uint max_index = 0, max_count = 0;
    for(auto it = discrete_mode_map.begin(); it != discrete_mode_map.end(); ++it) {
      if(it->second > max_count) {
	max_count = it->second;
	max_index = it->first;
      }
    }

    prediction.discrete_value_index = max_index;
  }

}


static bool find_nearest_neighbors_for_instance(const ml_instance_definition &mlid, 
						const ml_data &mld, const ml_instance &instance, 
						ml_uint k, ml_uint index_of_feature_to_predict, 
						ml_feature_value &prediction, 
						ml_vector<knn_neighbor> &neighbors_considered) {
  
  neighbors_considered.clear();

  if(instance.size() != mlid.size()) {
    log_error("knn: mismatch b/t instance and training instance sizes...\n");
    return(false);
  }
  
  ml_vector<knn_neighbor> all_distances;
  
  for(std::size_t ii = 0; ii < mld.size(); ++ii) {

    ml_double dist = 0;
    const ml_instance &neighbor = *mld[ii];

    for(std::size_t findex=0; findex < mlid.size(); ++findex) {
  
      if(findex == index_of_feature_to_predict) {
	continue;
      }

      if(mlid[findex]->type == ml_feature_type::discrete) {
	continue;
      }

      if(mlid[findex]->sd > 0.0) {
        ml_double norm_neighbor = (neighbor[findex].continuous_value - mlid[findex]->mean) / mlid[findex]->sd;
        ml_double norm_inst = (instance[findex].continuous_value - mlid[findex]->mean) / mlid[findex]->sd;
        ml_double diff = norm_inst - norm_neighbor;
        dist += (diff * diff);
      }
    }

    all_distances.push_back(std::make_pair(dist, mld[ii]));
  }

  std::sort(all_distances.begin(), all_distances.end());

  predict_using_neighbors(all_distances, mlid, k, 
			  index_of_feature_to_predict, 
			  prediction, neighbors_considered);

  return(true);
} 


bool knn::train(const ml_data &mld) {
  training_data_ = mld;
  validated_ = validate_input(mlid_, training_data_, 
			      k_, index_of_feature_to_predict_);
  if(!validated_) {
    return(false);
  }

  return(true);
}


ml_feature_value knn::evaluate(const ml_instance &instance, 
			       ml_vector<knn_neighbor> &neighbors) const {
  neighbors.clear();
  ml_feature_value prediction = {};
  if(!validated_) {
    return(prediction);
  }

  find_nearest_neighbors_for_instance(mlid_, training_data_,
				      instance, k_, 
				      index_of_feature_to_predict_,
				      prediction, neighbors);
  return(prediction);
}


ml_feature_value knn::evaluate(const ml_instance &instance) const {
  ml_vector<knn_neighbor> neighbors;
  ml_feature_value prediction = evaluate(instance, neighbors);
  return(prediction);
}


ml_string knn::summary() const {
  ml_string desc = "\n\n*** kNN Summary ***\n\n";
  desc += "k = " + std::to_string(k_);
  return(desc);
}

} // namespace puml
