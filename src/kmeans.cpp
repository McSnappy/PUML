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
#include "mlutil.h"

#include <algorithm>
#include <chrono>
#include <string.h>


namespace puml {

static const ml_string &KMEANS_BASEINFO_FILE = "kmeans.json";
static const ml_string &KMEANS_MLID_FILE = "mlid.json";


kmeans::kmeans(const ml_instance_definition &mlid, 
	       const ml_vector<ml_float> &feature_weights,
	       ml_uint k, ml_uint seed) :
  mlid_(mlid),
  feature_weights_(feature_weights),
  k_(k),
  seed_(seed) {
  
}


static bool validate_input(ml_uint k, 
			   const ml_instance_definition &mlid, 
			   const ml_data &mld, 
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

  const auto &inst_ptr = mld[0];
  if((*inst_ptr).size() != mlid.size()) {
    log_error("kmeans: mismatch b/t size of instance definition and instance\n");
    return(false);
  }

  for(std::size_t ii=0; ii < mlid.size(); ++ii) {

    if(feature_weights[ii] < 0.0) {
      log_error("kmeans: feature %d given negative weight...\n", ii);
      return(false);
    }

    if((feature_weights[ii] > 0.0) && (mlid[ii]->type != ml_feature_type::continuous)) {
      log_error("kmeans: '%s' is not a continuous feature (and has a nonzero weight)...\n", mlid[ii]->name.c_str());
      return(false);
    }

  }

  return(true);
}


static void init_kmeans_result(ml_uint k, const ml_instance_definition &mlid, 
			       ml_vector<kmeans_cluster> &clusters) {

  clusters.clear();

  for(std::size_t ii=0; ii < k; ++ii) {
    kmeans_cluster cluster;
    cluster.id = ii+1; // clusters 1 through k
    cluster.centroid.resize(mlid.size());
    clusters.push_back(cluster);
  }
}


static void init_cluster_ids(ml_uint k, ml_rng &rng, ml_vector<ml_uint> &mld_cluster_ids) {
  // Random parition initialization
  for(std::size_t ii=0; ii < mld_cluster_ids.size(); ++ii) {
    mld_cluster_ids[ii] = (rng.random_number() % k) + 1; // cluster ids from 1 to k
  }
}


static void update_centroids(const ml_instance_definition &mlid, const ml_data &mld,
			     const ml_vector<ml_float> &feature_weights, 
			     ml_vector<kmeans_cluster> &clusters,
			     ml_vector<ml_uint> &cluster_ids) {

  //
  // Find the centroid of each cluster given current instance cluster assignment
  //

  // reset cluster centroids and instance counts to zero
  for(std::size_t cluster_id=0; cluster_id < clusters.size(); ++cluster_id) {
    ml_vector<ml_double> &centroid = clusters[cluster_id].centroid;
    std::fill(centroid.begin(), centroid.end(), 0);
    clusters[cluster_id].instances = 0;
  }

  // update centroids using features with nonzero weights
  for(std::size_t instance_id=0; instance_id < mld.size(); ++instance_id) {
    const ml_instance &inst = *mld[instance_id];
    ml_uint cluster_id = cluster_ids[instance_id] - 1; // cluster ids from 1 to k
    clusters[cluster_id].instances += 1;

    for(std::size_t feature_id=0; feature_id < feature_weights.size(); ++feature_id) {
      if((feature_weights[feature_id] > 0.0) &&
	 (inst[feature_id].continuous_value != MISSING_CONTINUOUS_FEATURE_VALUE)) {
	clusters[cluster_id].centroid[feature_id] += (mlid[feature_id]->sd > 0.0) ? ((inst[feature_id].continuous_value - mlid[feature_id]->mean) / mlid[feature_id]->sd) : 0.0;
      }
    }
  }

  // divide each feature in the centroid by the number of instances in the cluster
  for(std::size_t cluster_id=0; cluster_id < clusters.size(); ++cluster_id) {
    ml_vector<ml_double> &centroid = clusters[cluster_id].centroid;
    std::transform(centroid.begin(), centroid.end(), centroid.begin(),
		   std::bind1st(std::multiplies<ml_double>(), (1.0 / clusters[cluster_id].instances)));
  }
}


static ml_uint _cluster_id_for_instance(const ml_instance_definition &mlid, const ml_instance &instance, 
					const ml_vector<ml_float> &feature_weights,
					const ml_vector<kmeans_cluster> &clusters, 
					ml_double &distsq) {

  // return the cluster id of the cluster centroid closest to the instance
  distsq = std::numeric_limits<ml_double>::max();
  ml_uint nearest_cluster_id = 1; // cluster ids from 1 to k

  for(std::size_t cluster_id=0; cluster_id < clusters.size(); ++cluster_id) {

    ml_double centroid_dist = 0;
    for(std::size_t feature_id=0; feature_id < feature_weights.size(); ++feature_id) {
      
      if((feature_weights[feature_id] > 0.0) &&
	 (instance[feature_id].continuous_value != MISSING_CONTINUOUS_FEATURE_VALUE)) {
	ml_double norm_feature = (mlid[feature_id]->sd > 0) ? ((instance[feature_id].continuous_value - mlid[feature_id]->mean) / mlid[feature_id]->sd) : 0.0;
	ml_double feature_dist = clusters[cluster_id].centroid[feature_id] - norm_feature;
	centroid_dist += (feature_weights[feature_id] * (feature_dist * feature_dist));
      }

    }

    if(centroid_dist < distsq) {
      distsq = centroid_dist;
      nearest_cluster_id = clusters[cluster_id].id;
    }
  }

  return(nearest_cluster_id);
}


static void assign_cluster_ids(const ml_instance_definition &mlid, 
			       const ml_data &mld, 
			       const ml_vector<ml_float> &feature_weights, 
			       const ml_vector<kmeans_cluster> &clusters,
			       ml_vector<ml_uint> &cluster_ids,
			       ml_double &rss) {
  
  //
  // Update instance cluster assignments given the current cluster centroids
  //
  rss = 0;

  ml_uint inst_index = 0;
  for(auto &inst_ptr : mld) {
    ml_double distsq = 0;
    cluster_ids[inst_index++] = _cluster_id_for_instance(mlid, *inst_ptr, 
					       feature_weights, 
					       clusters, distsq);
    rss += distsq;
  }
}


static bool _cluster_by_kmeans(ml_uint k, ml_uint seed, const ml_instance_definition &mlid, const ml_data &mld, 
			       const ml_vector<ml_float> &feature_weights, ml_vector<kmeans_cluster> &clusters,
			       ml_vector<ml_uint> &cluster_ids, ml_double &rss) {

  init_kmeans_result(k, mlid, clusters);

  ml_rng rng{seed};
  cluster_ids.clear();
  cluster_ids.resize(mld.size());
  init_cluster_ids(k, rng, cluster_ids);

  const ml_uint MAX_KMEANS_ITER = 10;
  for(ml_uint iter=0; iter < MAX_KMEANS_ITER; ++iter) {
    update_centroids(mlid, mld, feature_weights, clusters, cluster_ids);
    assign_cluster_ids(mlid, mld, feature_weights, clusters, cluster_ids, rss);
    // TODO: early stopping
  }
  
  return(true);
}


bool kmeans::cluster_by_kmeans(const ml_data &mld,
			       ml_vector<ml_uint> &mld_cluster_ids) {

  //
  // cluster KMEANS_TRIALS times with different seeds and select the clustering with lowest residual sum of squared error
  //
  const int KMEANS_TRIALS = 3;

  if(!validate_input(k_, mlid_, mld, feature_weights_)) {
    return(false);
  }
  
  ml_vector<kmeans_cluster> best_clusters;
  ml_double best_rss = std::numeric_limits<ml_double>::max();

  auto t1 = std::chrono::high_resolution_clock::now();
  for(int ii=0; ii < KMEANS_TRIALS; ++ii) {
    ml_vector<ml_uint> cluster_ids;
    if(!_cluster_by_kmeans(k_, seed_+ii, mlid_, mld, 
			   feature_weights_, clusters_, 
			   cluster_ids, rss_)) {
      return(false);
    }

    //puml::log("kmeans rss = %.3f\n", trial.rss);

    if(rss_ < best_rss) {
      mld_cluster_ids = cluster_ids;
      best_clusters = clusters_;
      best_rss = rss_;
    }
  }

  clusters_ = best_clusters;
  rss_ = best_rss;

  auto t2 = std::chrono::high_resolution_clock::now();
  ml_uint ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  log("k-means clustering took %.3f seconds (%u clusters, %u instances)\n", 
      (ms / 1000.0), k_, mld.size()); 

  return(true);
}


bool kmeans::cluster_by_kmeans(const ml_data &mld) {
  ml_vector<ml_uint> cluster_ids;
  return(cluster_by_kmeans(mld, cluster_ids));
}


static void add_kmeans_cluster_feature_to_inst_def(ml_instance_definition &mlid, 
						   const ml_vector<kmeans_cluster> &clusters,
						   const ml_string &feature_name) {
  
  //
  // Add a categorical feature to the instance definition for the k-means result.
  // The cluster id is 1 to k since we use 0 to represent the "unknown" category.
  //
  ml_feature_desc_ptr cdesc = std::make_shared<ml_feature_desc>();
  cdesc->name = feature_name;
  cdesc->type = ml_feature_type::discrete;
  cdesc->discrete_values.push_back(ML_UNKNOWN_DISCRETE_CATEGORY);
  cdesc->discrete_values_map[ML_UNKNOWN_DISCRETE_CATEGORY] = 0;
  cdesc->discrete_values_count.push_back(0);

  ml_uint mode_index = 0, mode_count = 0;
  for(std::size_t cluster_id=0; cluster_id < clusters.size(); ++cluster_id) {
    ml_uint cluster_id_cat = clusters[cluster_id].id; 
    ml_string cluster_category_name = std::to_string(cluster_id_cat);
    ml_uint cluster_instances = clusters[cluster_id].instances;
    cdesc->discrete_values.push_back(cluster_category_name);
    cdesc->discrete_values_map[cluster_category_name] = cluster_id_cat;
    cdesc->discrete_values_count.push_back(cluster_instances);
    if(cluster_instances > mode_count) {
      mode_count = cluster_instances;
      mode_index = cluster_id_cat;
    }
  }

  cdesc->discrete_mode_index = mode_index;
  mlid.push_back(cdesc);
}


static void add_kmeans_cluster_feature_to_data(const kmeans &kmeans,
					       ml_data &mld) {

  //
  // Add a ml_feature_value to each instance that represent the cluster id of centroid nearest the instance
  //
  for(const auto &inst_ptr : mld) {
    ml_feature_value cluster_id;
    cluster_id.discrete_value_index = kmeans.cluster_id_for_instance(*inst_ptr);
    (*inst_ptr).push_back(cluster_id);
  }

}


bool kmeans::add_kmeans_cluster_feature(ml_instance_definition &mlid, 
					ml_data &mld, const ml_string &feature_name) const {
  
  add_kmeans_cluster_feature_to_data(*this, mld);
  add_kmeans_cluster_feature_to_inst_def(mlid, clusters_, feature_name);

  return(true);
}


ml_uint kmeans::cluster_id_for_instance(const ml_instance &instance) const {
  ml_double distsq = 0;
  return(_cluster_id_for_instance(mlid_, instance, 
				  feature_weights_, 
				  clusters_,
				  distsq));
}

  
static cJSON *create_json_from_kmeans_result(const ml_vector<ml_float> &feature_weights,
					     const ml_vector<kmeans_cluster> &clusters,
					     ml_uint k, ml_uint seed, ml_uint rss) {

  cJSON *json_kmeans = cJSON_CreateObject();
  cJSON_AddStringToObject(json_kmeans, "object", "kmeans");

  cJSON_AddNumberToObject(json_kmeans, "k", k);
  cJSON_AddNumberToObject(json_kmeans, "seed", seed);
  cJSON_AddNumberToObject(json_kmeans, "rss", rss);

  // feature weights
  cJSON *json_feature_weights = cJSON_CreateArray();
  cJSON_AddItemToObject(json_kmeans, "feature_weights", json_feature_weights);
  for(std::size_t ii=0; ii < feature_weights.size(); ++ii) {
    cJSON_AddItemToArray(json_feature_weights, cJSON_CreateNumber(feature_weights[ii]));
  }

  // clusters
  cJSON *json_clusters = cJSON_CreateArray();
  cJSON_AddItemToObject(json_kmeans, "clusters", json_clusters);
  for(std::size_t cluster_id=0; cluster_id < clusters.size(); ++cluster_id) {
    
    const kmeans_cluster &cluster = clusters[cluster_id];
    
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


static bool get_feature_weights_from_json(cJSON *json_object, ml_vector<ml_float> &feature_weights) {

  feature_weights.clear();

  cJSON *weights_array = cJSON_GetObjectItem(json_object, "feature_weights");
  if(!weights_array || (weights_array->type != cJSON_Array)) {
    log_error("kmeans result json object is missing the weights array\n");
    return(false);
  }

  int weights_count = cJSON_GetArraySize(weights_array);
  for(int ii=0; ii < weights_count; ++ii) {
    cJSON *weight = cJSON_GetArrayItem(weights_array, ii);
    if(!weight || (weight->type != cJSON_Number)) {
      log_error("non-numeric kmeans weight...\n");
      return(false);
    }

    feature_weights.push_back(weight->valuedouble);
  }

  return(true);
}


static bool get_clusters_from_json(cJSON *json_object, ml_vector<kmeans_cluster> &clusters) {

  clusters.clear();

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
    if(!get_numeric_value_from_json(cluster_obj, "id", cluster.id)) {
      log_error("kmeans result json cluster object has invalid id\n");
      return(false);
    }
    
    // cluster instance count
    if(!get_numeric_value_from_json(cluster_obj, "instances", cluster.instances)) {
      log_error("kmeans result json cluster object has invalid instances count\n");
      return(false);
    }
    
    // cluster centroid
    cJSON *centroid_array = cJSON_GetObjectItem(cluster_obj, "centroid");
    if(!centroid_array || (centroid_array->type != cJSON_Array)) {
      log_error("kmeans result json cluster object has invalid centroid object\n");
      return(false);
    }

    int feature_count = cJSON_GetArraySize(centroid_array);
    for(int feature_id=0; feature_id < feature_count; ++feature_id) {
      cJSON *feature = cJSON_GetArrayItem(centroid_array, feature_id);
      if(!feature || (feature->type != cJSON_Number)) {
	log_error("non-numeric kmeans centroid...\n");
	return(false);
      }

      cluster.centroid.push_back(feature->valuedouble);
    }

    clusters.push_back(cluster);
  }

  return(true);
}


bool kmeans::read_kmeans_from_file(const ml_string &path) {

  cJSON *json_object = read_model_json_from_file(path);
  if(!json_object) {
    log_error("couldn't load k-means result json object from file: %s\n", path.c_str());
    return(false);
  }
  
  cJSON *object = cJSON_GetObjectItem(json_object, "object");
  if(!object || !object->valuestring || strcmp(object->valuestring, "kmeans")) {
    log_error("json object is not a kmeans result...\n");
    cJSON_Delete(json_object);
    return(false);
  }

  bool status = true;
  if(!(get_double_value_from_json(json_object, "rss", rss_) &&
       get_numeric_value_from_json(json_object, "k", k_) &&
       get_numeric_value_from_json(json_object, "seed", seed_) &&
       get_feature_weights_from_json(json_object, feature_weights_) &&
       get_clusters_from_json(json_object, clusters_))) {
     status = false;
  }
       
  cJSON_Delete(json_object);

  return(status);
}


bool kmeans::restore(const ml_string &path) {

  seed_ = 0;
  rss_ = 0;
  k_ = 0;
  feature_weights_.clear();
  clusters_.clear();

  if(!read_instance_definition_from_file(path + "/" + KMEANS_MLID_FILE, mlid_)) {
    log_error("couldn't read kmeans instance defintion\n");
    return(false);
  }

  if(!read_kmeans_from_file(path + "/" + KMEANS_BASEINFO_FILE)) {
    return(false);
  }

  return(true);
}


bool kmeans::save(const ml_string &path) const {

  if(mlid_.empty()) {
    return(false);
  }

  if(!prepare_directory_for_model_save(path)) {
    return(false);
  }

  // write the instance definition
  if(!write_instance_definition_to_file(path + "/" + KMEANS_MLID_FILE, mlid_)) {
    log_error("couldn't write kmeans instance definition to %s\n", KMEANS_MLID_FILE.c_str());
    return(false);
  }

  // write the kmeans result json
  cJSON *json_object = create_json_from_kmeans_result(feature_weights_,
						      clusters_,
						      k_, seed_, rss_);
  if(!json_object) {
    log_error("couldn't create json object from k-means result\n");
    return(false);
  }

  bool status = write_model_json_to_file(path + "/" + KMEANS_BASEINFO_FILE, json_object);
  cJSON_Delete(json_object);

  return(status);  
}

} // namespace puml
