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

#include "kmeans.h"
#include <algorithm>
#include <chrono>
#include <string.h>


namespace puml {
  
static bool validateInput(ml_uint k, const ml_instance_definition &mlid, const ml_data &mld, 
			  const ml_vector<ml_float> &feature_weights) {

  if(k == 0) {
    log_error("kmeans: k must be > 0\n");
    return(false);
  }
			
  if(mlid.empty()) {
    log_error("kmeans: empty instance definition...\n");
    return(false);
  }

  if(mld.empty()) {
    log_error("kmeans: empty mld...\n");
    return(false);
  }

  if(mlid.size() != feature_weights.size()) {
    log_error("kmeans: mismatch b/t size of instance definition and # of feature weights\n");
    return(false);
  }

  for(std::size_t ii=0; ii < mlid.size(); ++ii) {

    if(feature_weights[ii] < 0.0) {
      log_error("kmeans: feature %d given negative weight...\n", ii);
      return(false);
    }

    if((feature_weights[ii] > 0.0) && (mlid[ii].type != ML_FEATURE_TYPE_CONTINUOUS)) {
      log_error("kmeans: '%s' is not a continuous feature (and has a nonzero weight)...\n", mlid[ii].name.c_str());
      return(false);
    }

  }

  return(true);
}


static void initkMeansResult(ml_uint k, const ml_instance_definition &mlid, 
			     const ml_vector<ml_float> &feature_weights, 
			     kmeans_result &result) {
  result.feature_weights = feature_weights;
  for(std::size_t ii=0; ii < k; ++ii) {
    kmeans_cluster cluster = {};
    cluster.id = ii+1; // clusters 1 through k
    cluster.centroid.resize(mlid.size());
    result.clusters.push_back(cluster);
  }
}


static void initClusterIds(ml_uint k, ml_rng_config *rng_config, ml_vector<ml_uint> &mld_cluster_ids) {
  // Random parition initialization
  for(std::size_t ii=0; ii < mld_cluster_ids.size(); ++ii) {
    mld_cluster_ids[ii] = (generateRandomNumber(rng_config) % k) + 1; // cluster ids from 1 to k
  }
}


static void updateCentroids(const ml_instance_definition &mlid, const ml_data &mld, 
			    kmeans_result &result,
			    const ml_vector<ml_float> &feature_weights, 
			    ml_vector<ml_uint> &cluster_ids) {

  //
  // Find the centroid of each cluster given current instance cluster assignment
  //

  // reset cluster centroids and instance counts to zero
  for(std::size_t cluster_id=0; cluster_id < result.clusters.size(); ++cluster_id) {
    ml_vector<ml_double> &centroid = result.clusters[cluster_id].centroid;
    std::fill(centroid.begin(), centroid.end(), 0);
    result.clusters[cluster_id].instances = 0;
  }

  // update centroids using features with nonzero weights
  for(std::size_t instance_id=0; instance_id < mld.size(); ++instance_id) {
    const ml_instance &inst = *mld[instance_id];
    ml_uint cluster_id = cluster_ids[instance_id] - 1; // cluster ids from 1 to k
    result.clusters[cluster_id].instances += 1;

    for(std::size_t feature_id=0; feature_id < feature_weights.size(); ++feature_id) {
      if(feature_weights[feature_id] > 0.0) {
	result.clusters[cluster_id].centroid[feature_id] += (mlid[feature_id].sd > 0.0) ? ((inst[feature_id].continuous_value - mlid[feature_id].mean) / mlid[feature_id].sd) : 0.0;
      }
    }
  }

  // divide each feature in the centroid by the number of instances in the cluster
  for(std::size_t cluster_id=0; cluster_id < result.clusters.size(); ++cluster_id) {
    ml_vector<ml_double> &centroid = result.clusters[cluster_id].centroid;
    std::transform(centroid.begin(), centroid.end(), centroid.begin(),
		   std::bind1st(std::multiplies<ml_double>(), (1.0 / result.clusters[cluster_id].instances)));
  }
}


static ml_uint _clusterIdForInstance(const ml_instance_definition &mlid, const ml_instance &instance, 
				     const kmeans_result &result, ml_double &distsq) {

  // return the cluster id of the cluster centroid closest to the instance
  distsq = std::numeric_limits<ml_double>::max();
  ml_uint nearest_cluster_id = 0;

  for(std::size_t cluster_id=0; cluster_id < result.clusters.size(); ++cluster_id) {

    ml_double centroid_dist = 0;
    for(std::size_t feature_id=0; feature_id < result.feature_weights.size(); ++feature_id) {
      
      if(result.feature_weights[feature_id] > 0.0) {
	ml_double norm_feature = (mlid[feature_id].sd > 0) ? ((instance[feature_id].continuous_value - mlid[feature_id].mean) / mlid[feature_id].sd) : 0.0;
	ml_double feature_dist = result.clusters[cluster_id].centroid[feature_id] - norm_feature;
	centroid_dist += (result.feature_weights[feature_id] * (feature_dist * feature_dist));
      }

    }

    if(centroid_dist < distsq) {
      distsq = centroid_dist;
      nearest_cluster_id = result.clusters[cluster_id].id;
    }
  }

  return(nearest_cluster_id);
}


static void assignClusterIds(const ml_instance_definition &mlid, const ml_data &mld, kmeans_result &result, ml_vector<ml_uint> &cluster_ids) {
  
  //
  // Update instance cluster assignments given the current cluster centroids
  //
  result.rss = 0;

  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    const ml_instance &inst = *mld[ii];
    ml_double distsq = 0;
    cluster_ids[ii] = _clusterIdForInstance(mlid, inst, result, distsq);
    result.rss += distsq;
  }
}


static bool _clusterBykMeans(ml_uint k, ml_uint seed, const ml_instance_definition &mlid, const ml_data &mld, 
			     const ml_vector<ml_float> &feature_weights, kmeans_result &result,
			     ml_vector<ml_uint> *mld_cluster_ids) {

  initkMeansResult(k, mlid, feature_weights, result);

  ml_rng_config *rng_config = createRngConfigWithSeed(seed);
  ml_vector<ml_uint> cluster_ids(mld.size());
  initClusterIds(k, rng_config, cluster_ids);

  const ml_uint MAX_KMEANS_ITER = 10;
  for(ml_uint iter=0; iter < MAX_KMEANS_ITER; ++iter) {
    updateCentroids(mlid, mld, result, feature_weights, cluster_ids);
    assignClusterIds(mlid, mld, result, cluster_ids);
    // TODO: early stopping
  }
  
  delete rng_config;

  if(mld_cluster_ids) {
    *mld_cluster_ids = cluster_ids;
  }

  return(true);
}


bool clusterBykMeans(ml_uint k, ml_uint seed, const ml_instance_definition &mlid, const ml_data &mld, 
		     const ml_vector<ml_float> &feature_weights, kmeans_result &result,    
		     ml_vector<ml_uint> *mld_cluster_ids) {

  //
  // cluster KMEANS_TRIALS times with different seeds and select the clustering with lowest residual sum of squared error
  //
  const int KMEANS_TRIALS = 3;

  if(!validateInput(k, mlid, mld, feature_weights)) {
    return(false);
  }
  
  result.rss = std::numeric_limits<ml_double>::max();
  auto t1 = std::chrono::high_resolution_clock::now();
  for(int ii=0; ii < KMEANS_TRIALS; ++ii) {
    kmeans_result trial = {};
    if(!_clusterBykMeans(k, seed+ii, mlid, mld, feature_weights, trial, mld_cluster_ids)) {
      return(false);
    }

    //puml::log("kmeans rss = %.3f\n", trial.rss);

    if(trial.rss < result.rss) {
      result = trial;
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  ml_uint ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  log("k-means clustering took %.3f seconds (%u clusters, %u instances)\n", (ms / 1000.0), k, mld.size()); 

  return(true);
}


static void addkMeansClusterFeatureInstanceDefinition(ml_instance_definition &mlid, const kmeans_result &result, const ml_string &name) {
  
  //
  // Add a categorical feature to the instance definition for the k-means result.
  // The cluster id is 1 to k since we use 0 to represent the "unknown" category.
  //
  ml_feature_desc cdesc = {};
  cdesc.name = name;
  cdesc.type = ML_FEATURE_TYPE_DISCRETE;
  cdesc.discrete_values.push_back(ML_UNKNOWN_DISCRETE_CATEGORY);
  cdesc.discrete_values_map[ML_UNKNOWN_DISCRETE_CATEGORY] = 0;
  cdesc.discrete_values_count.push_back(0);

  ml_uint mode_index = 0, mode_count = 0;
  for(std::size_t cluster_id=0; cluster_id < result.clusters.size(); ++cluster_id) {
    ml_uint cluster_id_cat = result.clusters[cluster_id].id; 
    ml_string cluster_category_name = std::to_string(cluster_id_cat);
    ml_uint cluster_instances = result.clusters[cluster_id].instances;
    cdesc.discrete_values.push_back(cluster_category_name);
    cdesc.discrete_values_map[cluster_category_name] = cluster_id_cat;
    cdesc.discrete_values_count.push_back(cluster_instances);
    if(cluster_instances > mode_count) {
      mode_count = cluster_instances;
      mode_index = cluster_id_cat;
    }
  }

  cdesc.discrete_mode_index = mode_index;
  mlid.push_back(cdesc);
}


static void addkMeansClusterFeatureData(const ml_instance_definition &mlid, ml_mutable_data &mld, const kmeans_result &result) {

  //
  // Add a ml_feature_value to each instance that represent the cluster id of centroid nearest the instance
  //
  for(std::size_t instance_id = 0; instance_id < mld.size(); ++instance_id) {
    ml_instance &inst = *mld[instance_id];
    ml_feature_value cluster_id;
    cluster_id.discrete_value_index = clusterIdForInstance(mlid, inst, result); 
    inst.push_back(cluster_id);
  }

}


bool addkMeansClusterFeature(ml_instance_definition &mlid, ml_mutable_data &mld, const kmeans_result &result, const ml_string &name) {
  
  addkMeansClusterFeatureData(mlid, mld, result);
  addkMeansClusterFeatureInstanceDefinition(mlid, result, name);

  return(true);
}


ml_uint clusterIdForInstance(const ml_instance_definition &mlid, const ml_instance &instance, const kmeans_result &result) {
  ml_double distsq = 0;
  return(_clusterIdForInstance(mlid, instance, result, distsq));
}


static cJSON *createJSONObjectFromkMeansResult(const kmeans_result &result) {
  cJSON *json_kmeans = cJSON_CreateObject();
  cJSON_AddStringToObject(json_kmeans, "object", "kmeans_result");

  // rss
  cJSON_AddNumberToObject(json_kmeans, "rss", result.rss);

  // feature weights
  cJSON *json_feature_weights = cJSON_CreateArray();
  cJSON_AddItemToObject(json_kmeans, "feature_weights", json_feature_weights);
  for(std::size_t ii=0; ii < result.feature_weights.size(); ++ii) {
    cJSON_AddItemToArray(json_feature_weights, cJSON_CreateNumber(result.feature_weights[ii]));
  }

  // clusters
  cJSON *json_clusters = cJSON_CreateArray();
  cJSON_AddItemToObject(json_kmeans, "clusters", json_clusters);
  for(std::size_t cluster_id=0; cluster_id < result.clusters.size(); ++cluster_id) {
    
    const kmeans_cluster &cluster = result.clusters[cluster_id];
    
    // cluster id & instance count
    cJSON *json_cluster_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(json_cluster_obj, "id", cluster.id);
    cJSON_AddNumberToObject(json_cluster_obj, "instances", cluster.instances);

    // cluster centroid
    cJSON *json_cluster_centroid = cJSON_CreateArray();
    cJSON_AddItemToObject(json_cluster_obj, "centroid", json_cluster_centroid);
    for(std::size_t feature_id=0; feature_id < cluster.centroid.size(); ++feature_id) {
      cJSON_AddItemToArray(json_cluster_centroid, cJSON_CreateNumber(cluster.centroid[feature_id]));
    }
    
    cJSON_AddItemToArray(json_clusters, json_cluster_obj);
  }

  return(json_kmeans);
}


bool writekMeansResultToFile(const ml_string &path_to_file, const kmeans_result &result) {
  cJSON *json_object = createJSONObjectFromkMeansResult(result);
  if(!json_object) {
    log_error("couldn't create json object from k-means result\n");
    return(false);
  }

  bool status = writeModelJSONToFile(path_to_file, json_object);
  cJSON_Delete(json_object);

  return(status);
}


static bool createkMeansResultFromJSONObject(cJSON *json_object, kmeans_result &result) {
  
  if(!json_object) {
    log_error("kmeans: nil json object\n");
    return(false);
  }

  cJSON *object = cJSON_GetObjectItem(json_object, "object");
  if(!object || !object->valuestring || strcmp(object->valuestring, "kmeans_result")) {
    log_error("json object is not a kmeans result...\n");
    return(false);
  }

  // rss
  cJSON *rss_obj = cJSON_GetObjectItem(json_object, "rss");
  result.rss = rss_obj->valuedouble;

  // feature weights
  cJSON *weights_array = cJSON_GetObjectItem(json_object, "feature_weights");
  if(!weights_array || (weights_array->type != cJSON_Array)) {
    log_error("kmeans result json object is missing the weights array\n");
    return(false);
  }

  int weights_count = cJSON_GetArraySize(weights_array);
  for(int ii=0; ii < weights_count; ++ii) {
    cJSON *weight = cJSON_GetArrayItem(weights_array, ii);
    result.feature_weights.push_back(weight->valuedouble);
  }

  // clusters
  cJSON *clusters_array = cJSON_GetObjectItem(json_object, "clusters");
  if(!clusters_array || (clusters_array->type != cJSON_Array)) {
    log_error("kmeans result json object is missing the clusters array\n");
    return(false);
  }
  
  int clusters_count = cJSON_GetArraySize(clusters_array);
  for(int ii=0; ii < clusters_count; ++ii) {
    kmeans_cluster cluster;
    cJSON *cluster_obj = cJSON_GetArrayItem(clusters_array, ii);

    // cluster id
    cJSON *id_obj = cJSON_GetObjectItem(cluster_obj, "id");
    if(!id_obj || (id_obj->type != cJSON_Number)) {
      log_error("kmeans result json cluster object has invalid id\n");
      return(false);
    }
    
    cluster.id = id_obj->valueint;

    // cluster instance count
    cJSON *instances_obj = cJSON_GetObjectItem(cluster_obj, "instances");
    if(!instances_obj || (instances_obj->type != cJSON_Number)) {
      log_error("kmeans result json cluster object has invalid instances count\n");
      return(false);
    }
    
    cluster.instances = instances_obj->valueint;

    // cluster centroid
    cJSON *centroid_array = cJSON_GetObjectItem(cluster_obj, "centroid");
    if(!centroid_array || (centroid_array->type != cJSON_Array)) {
      log_error("kmeans result json cluster object has invalid centroid object\n");
      return(false);
    }

    int feature_count = cJSON_GetArraySize(centroid_array);
    for(int feature_id=0; feature_id < feature_count; ++feature_id) {
      cJSON *feature = cJSON_GetArrayItem(centroid_array, feature_id);
      cluster.centroid.push_back(feature->valuedouble);
    }

    result.clusters.push_back(cluster);
  }

  return(true);
}


bool readkMeansResultFromFile(const ml_string &path_to_file, kmeans_result &result) {
  
  result.rss = 0;
  result.feature_weights.clear();
  result.clusters.clear();
  
  cJSON *json_object = readModelJSONFromFile(path_to_file);
  if(!json_object) {
    log_error("couldn't load k-means result json object from file: %s\n", path_to_file.c_str());
    return(false);
  }

  bool status = createkMeansResultFromJSONObject(json_object, result);
  cJSON_Delete(json_object);

  return(status);
}


} // namespace puml
