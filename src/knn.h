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

#include "machinelearning.h"

namespace puml {

  typedef std::pair<ml_double, ml_instance_ptr> knn_neighbor; // distance and the instance 
  
  bool findNearestNeighborsForInstance(const ml_instance_definition &mlid, const ml_data &mld, const ml_instance &instance, 
				       ml_uint k, ml_uint index_of_feature_to_predict, ml_feature_value &prediction, 
				       ml_vector<knn_neighbor> *neighbors_considered = nullptr);
  
  bool printNearestNeighborsResultsForData(const ml_instance_definition &mlid, const ml_data &training,
					   const ml_data &test, ml_uint k, ml_uint index_of_feature_to_predict);

} 

