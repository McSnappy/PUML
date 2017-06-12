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

  struct kmeans_cluster {
    ml_uint id = 0;
    ml_uint instances = 0;
    ml_vector<ml_double> centroid;
  };


  class kmeans {

  public:

    kmeans(const ml_string &path) { restore(path); }

    kmeans(const ml_instance_definition &mlid, 
	   const ml_vector<ml_float> &feature_weights,
	   ml_uint k, ml_uint seed = ML_DEFAULT_SEED);

    bool save(const ml_string &path) const;
    bool restore(const ml_string &path);


    //
    // Find k clusters using a random initial partition and feature weighting. 
    // mld_cluster_ids is an optional parameter that will be populated with the 
    // cluster ids found for each instance in mld.
    //
    // returns true on success
    //
    bool cluster_by_kmeans(const ml_data &mld);
 
    bool cluster_by_kmeans(const ml_data &mld, 
			   ml_vector<ml_uint> &mld_cluster_ids);
    


    //
    // Returns the cluster id (1 to k) of the cluster with centroid nearest to instance
    //
    ml_uint cluster_id_for_instance(const ml_instance &instance) const;


    //
    // Add the k-means clustering as a feature to mlid & mld.
    // 
    // returns true on success
    //
    bool add_kmeans_cluster_feature(ml_instance_definition &mlid, 
				    ml_data &mld, const ml_string &feature_name) const;
    

    const ml_instance_definition &mlid() const { return(mlid_); }
    const ml_uint k() const { return(k_); }
    const ml_uint seed() const { return(seed_); }

  private:

    ml_instance_definition mlid_;
    ml_vector<ml_float> feature_weights_;
    ml_uint k_ = 0;
    ml_uint seed_ = ML_DEFAULT_SEED;
    ml_vector<kmeans_cluster> clusters_;
    ml_double rss_ = 0.0;

    // implementation
    bool read_kmeans_from_file(const ml_string &path);
  };


} 


