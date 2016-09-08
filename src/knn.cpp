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

namespace puml {

static bool validateInput(const ml_instance_definition &mlid, const ml_data &mld, 
			  const ml_instance &instance, ml_uint k, ml_uint index_of_feature_to_predict) {
			
  if(mlid.empty()) {
    log_error("empty instance definition...\n");
    return(false);
  }

  if(mld.empty()) {
    log_error("empty neighbor list...\n");
    return(false);
  }

  if(instance.size() != mlid.size()) {
    log_error("mismatch b/t instance and definition sizes...\n");
    return(false);
  }

  if(index_of_feature_to_predict >= mlid.size()) {
    log_error("invalid feature to predict...\n");
    return(false);
  }

  for(std::size_t ii=0; ii < mlid.size(); ++ii) {

    if(ii == index_of_feature_to_predict) {
      continue;
    }

    if(mlid[ii].type != ML_FEATURE_TYPE_CONTINUOUS) {
      log_error("'%s' is not a continuous feature...\n", mlid[ii].name.c_str());
      return(false);
    }
  }

  if(k == 0) {
    log_error("k must be > 0\n");
    return(false);
  }

  return(true);
}

static void predictUsingNeighbors(const ml_vector<knn_neighbor> &all_distances, const ml_instance_definition &mlid, 
				  ml_uint k, ml_uint index_of_feature_to_predict, ml_feature_value &prediction, 
				  ml_vector<knn_neighbor> *neighbors_considered) {

  ml_double continuous_sum = 0;
  ml_uint continuous_count = 0;
  ml_map<ml_uint, ml_uint> discrete_mode_map;

  for(std::size_t dist_index = 0; (dist_index < k) && (dist_index < all_distances.size()); ++dist_index) {
    
    const ml_instance *neighbor = all_distances[dist_index].second;
    
    if(neighbors_considered) {
      neighbors_considered->push_back(all_distances[dist_index]);
    }

    if(mlid[index_of_feature_to_predict].type == ML_FEATURE_TYPE_CONTINUOUS) {
      continuous_sum += (*neighbor)[index_of_feature_to_predict].continuous_value;
      continuous_count += 1;
    }
    else {
      discrete_mode_map[(*neighbor)[index_of_feature_to_predict].discrete_value_index] += 1;
    }

  }

  if(mlid[index_of_feature_to_predict].type == ML_FEATURE_TYPE_CONTINUOUS) {
    prediction.continuous_value = (continuous_sum / continuous_count);
  }
  else {
    ml_uint max_index = 0, max_count = 0;
    for(ml_map<ml_uint, ml_uint>::iterator it = discrete_mode_map.begin(); it != discrete_mode_map.end(); ++it) {
      if(it->second > max_count) {
	max_count = it->second;
	max_index = it->first;
      }
    }

    prediction.discrete_value_index = max_index;
  }

}

bool findNearestNeighborsForInstance(const ml_instance_definition &mlid, const ml_data &mld, const ml_instance &instance, 
				     ml_uint k, ml_uint index_of_feature_to_predict, ml_feature_value &prediction, 
				     ml_vector<knn_neighbor> *neighbors_considered) {
  
  if(neighbors_considered) {
    neighbors_considered->clear();
  }

  if(!validateInput(mlid, mld, instance, k, index_of_feature_to_predict)) {
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

      if(mlid[findex].sd > 0.0) {
        // TODO: mld should be normalized once
        ml_double norm_neighbor = (neighbor[findex].continuous_value - mlid[findex].mean) / mlid[findex].sd;
        ml_double norm_inst = (instance[findex].continuous_value - mlid[findex].mean) / mlid[findex].sd;
        ml_double diff = norm_inst - norm_neighbor;
        dist += (diff * diff);
      }
    }

    all_distances.push_back(std::make_pair(dist, mld[ii]));
  }

  std::sort(all_distances.begin(), all_distances.end());

  predictUsingNeighbors(all_distances, mlid, k, index_of_feature_to_predict, prediction, neighbors_considered);

  return(true);
} 

bool printNearestNeighborsResultsForData(const ml_instance_definition &mlid, const ml_data &training,
					 const ml_data &test, ml_uint k, ml_uint index_of_feature_to_predict) {
  ml_classification_results cr = {};
  ml_regression_results rr = {};

  for(std::size_t ii = 0; ii < test.size(); ++ii) {

    ml_feature_value prediction;
    if(!findNearestNeighborsForInstance(mlid, training, *test[ii], k, index_of_feature_to_predict, prediction)) {
      return(false);
    }

    if(mlid[index_of_feature_to_predict].type == ML_FEATURE_TYPE_CONTINUOUS) {
      collectRegressionResultForInstance(mlid, index_of_feature_to_predict, *test[ii], &prediction, rr);
    }
    else {
      collectClassificationResultForInstance(mlid, index_of_feature_to_predict, *test[ii], &prediction, cr);
    }
  }


  if(mlid[index_of_feature_to_predict].type == ML_FEATURE_TYPE_CONTINUOUS) {
    printRegressionResultsSummary(rr);
  }
  else {
    printClassificationResultsSummary(mlid, index_of_feature_to_predict, cr);
  }

  return(true);
}


} // namespace puml
