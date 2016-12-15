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

#ifndef __KMEANS_H__
#define __KMEANS_H__

#include "machinelearning.h"

namespace puml {

  typedef struct {
    ml_uint id;
    ml_uint instances;
    ml_vector<ml_double> centroid;
  } kmeans_cluster;


  typedef struct {
    ml_vector<kmeans_cluster> clusters;
    ml_vector<ml_float> feature_weights;
    ml_double rss;
  } kmeans_result;


  //
  // Find k clusters using a random initial partition and feature weighting. 
  // mld_cluster_ids is an optional parameter that will be populated with the 
  // cluster ids found for each instance in mld.
  //
  // returns true on success
  // 
  bool clusterBykMeans(ml_uint k, ml_uint seed, const ml_instance_definition &mlid, const ml_data &mld, 
		       const ml_vector<ml_float> &feature_weights, kmeans_result &result,
		       ml_vector<ml_uint> *mld_cluster_ids = nullptr);

  //
  // Add the k-means clustering as a feature to mld.
  // 
  // returns true on success
  //
  bool addkMeansClusterFeature(ml_instance_definition &mlid, ml_mutable_data &mld, const kmeans_result &result, const ml_string &name);

  //
  // Returns the cluster id (1 to k) of the cluster with centroid nearest to instance
  //
  ml_uint clusterIdForInstance(const ml_instance_definition &mlid, const ml_instance &instance, const kmeans_result &result);
  
  //
  // Read/Write k-means result to disk (JSON)
  //
  bool writekMeansResultToFile(const ml_string &path_to_file, const kmeans_result &result);
  bool readkMeansResultFromFile(const ml_string &path_to_file, kmeans_result &result);

} 

#endif
