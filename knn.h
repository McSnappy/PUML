#ifndef __KNN_H__
#define __KNN_H__

#include "machinelearning.h"

typedef std::pair<ml_double, const ml_instance *> knn_neighbor; // distance and the instance 

bool knn_findNearestNeighborsForInstance(const ml_instance_definition &mlid, const ml_data &mld, const ml_instance &instance, 
					 ml_uint k, ml_uint index_of_feature_to_predict, ml_feature_value &prediction, 
					 ml_vector<knn_neighbor> *neighbors_considered = nullptr);

bool knn_printNearestNeighborsResultsForData(const ml_instance_definition &mlid, const ml_data &training,
					     const ml_data &test, ml_uint k, ml_uint index_of_feature_to_predict);


#endif
