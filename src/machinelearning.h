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

#include <stdint.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>

#include "logging.h"
#include "cJSON/cJSON.h"

namespace puml {

typedef enum {

  ML_FEATURE_TYPE_CONTINUOUS = 0,
  ML_FEATURE_TYPE_DISCRETE // Categorical

} ml_feature_type;


// 
// ml_float is used to represent continuous feature values.
// ml_double is used when we operate on continuous feature values.
// see ml_feature_value below
//
using ml_float = float;
using ml_double = double;
using ml_uint = uint32_t;
using ml_string = std::string;

extern const ml_float ML_VERSION; 
extern const ml_float MISSING_CONTINUOUS_FEATURE_VALUE;

template <typename T>
using ml_vector = std::vector<T>;

template <typename T, typename T2>
using ml_map = std::unordered_map<T, T2>;


//
// Random Number Generator
//
using ml_rng = std::mt19937;

struct ml_rng_config {
  ml_rng rng;
};


// pass a 0 to initialize the seed using the clock
ml_rng_config *createRngConfigWithSeed(ml_uint seed); 
ml_uint generateRandomNumber(ml_rng_config *rng_config);


//
// we need our own shuffle since std::shuffle isn't guaranteed
// to give identical results across machines and compilers
//
template <typename T>
void shuffleVector(ml_vector<T> &vec, ml_rng_config *rng_config);

template <typename T>
void shuffleVector(ml_vector<T> &vec, ml_rng_config *rng_config) {
  for (std::size_t ii = vec.size() - 1; ii > 0; --ii) {
    std::swap(vec[ii], vec[generateRandomNumber(rng_config) % (ii + 1)]);
  }
}


// 
// make_unique for c++11
//
template<typename T, typename ...Args>
std::unique_ptr<T> make_unique( Args&& ...args ) {
  return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
}


//
// ml_feature_desc holds info about a feature's name, type, distribution, etc
//
struct ml_feature_desc {

  ml_feature_type type;
  ml_string name;
  ml_uint missing;
  bool preserve_missing; // false (default): use feature's global mean/mode. 
                         // true: insert out of range value for continuous features and separate category for discrete. 
                         // This option is available from the instance defintion row, the first row, of the data set. 
                         // see loadInstanceDataFromFile() below

  ml_float mean;
  ml_float sd;

  ml_vector<ml_string> discrete_values;
  ml_map<ml_string, ml_uint> discrete_values_map;
  ml_vector<ml_uint> discrete_values_count;
  ml_uint discrete_mode_index;

};


//
// change the typedef for ml_float above to alter
// the precision for continuous features and storage 
// requirements for all features.
//
union ml_feature_value {

  ml_float continuous_value;
  ml_uint discrete_value_index;   
   
};


// 
// an ml_instance_definition is a vector of ml_feature_desc, one
// to describe each feature column type, name, distribution, etc.
//
using ml_instance_definition = ml_vector<ml_feature_desc>;


//
// ml_instance is a vector of feature values, parallel to some
// ml_instance_definition
//
using ml_instance = ml_vector<ml_feature_value>;


//
// a vector of instances represents a dataset
//
using ml_instance_ptr = std::shared_ptr<ml_instance>;
using ml_data =  ml_vector<ml_instance_ptr>;


//
// loadInstanceDataFromFile() 
//
// path_to_input_file -- csv file where each row represents an instance, and the first row is in 
//                       the instance definition format below.
// ml_instance_definition -- will be populated with the features defined by the first row
// ml_data -- will be populated with instance data from the csv
// 
// returns true on success
//
// Instance Definition Row:
// Name:Type:Optional, for example Feature1:C for a continuous feature, or
// SomeFeature:D for a discrete (categorical) feature, or Feature:I to ignore.
// 
// You can specify Feature1:C:P or Feature1:D:P to preserve any missing values.
// With :P, an out of range value will be used for missing continuous features,
// and a separate category for missing discrete features will be created. The 
// default will use the feature's global mean or mode to populate missing values.
//
bool loadInstanceDataFromFile(const ml_string &path_to_input_file, ml_instance_definition &mlid, ml_data &mld);


//
// The internal category values for discrete features are defined based on the order that each
// distinct category is found in the data when loaded with loadInstanceDataFromFile(). If you
// wish to run a model built from training data on test data from a separate file you should
// use loadInstanceDataFromFileUsingInstanceDefinition() to force the load of test data 
// to use category definitions from the training data's ml_instance_definition.
// 
// returns true on success
//
// ids is an optional parameter that will be populated with the first column of the data file (id column
// of test data from kaggle competition, etc).
//
bool loadInstanceDataFromFileUsingInstanceDefinition(const ml_string &path_to_input_file, const ml_instance_definition &mlid, 
						     ml_data &mld, ml_vector<ml_string> *ids = nullptr);


//
// populate training and test vectors with instances randomly chosen from mld.
// after the call, training will have training_factor fraction of the data, test will have 
// the remainder and mld will be empty.
//
void splitDataIntoTrainingAndTest(ml_data &mld, ml_float training_factor, 
				  ml_rng_config *rng_config, ml_data &training, 
				  ml_data &test);


//
// Displays a summary of features including name, type, distribution, etc.
//
void printInstanceDataSummary(const ml_instance_definition &mlid);


//
// Returns the internal column index for a feature with the given name
//
ml_uint indexOfFeatureWithName(const ml_string &feature_name, const ml_instance_definition &mlid);


//
// One Hot Encoding: Converts discrete (categorical) features to continuous binary features
//
bool createOneHotEncodingForData(const ml_instance_definition &mlid, const ml_data &mld, 
				 const ml_string &name_of_feature_to_predict, 
				 ml_instance_definition &mlid_ohe, ml_data &mld_ohe);


//
// Save/Read Instance Definition To Disk (JSON)
//
bool writeInstanceDefinitionToFile(const ml_string &path_to_file, const ml_instance_definition &mlid);
bool readInstanceDefinitionFromFile(const ml_string &path_to_file, ml_instance_definition &mlid);


// 
// Aggregate Classification Results
//
struct ml_classification_results {

  ml_uint instances;
  ml_uint instances_correctly_classified;
  ml_map<ml_string, ml_uint> confusion_matrix_map;

};


void collectClassificationResultForInstance(const ml_instance_definition &mlid, ml_uint index_of_feature_to_predict, const ml_instance &instance, 
					    const ml_feature_value *result, ml_classification_results &mlcr);
void printClassificationResultsSummary(const ml_instance_definition &mlid, ml_uint index_of_feature_to_predict, const ml_classification_results &mlcr);
 

//
// Aggregate Regression Results
//
struct ml_regression_results {

  ml_uint instances;
  ml_double sum_absolute_error;
  ml_double sum_mean_squared_error;
  //ml_double sum_log_loss;
  
};

void collectRegressionResultForInstance(const ml_instance_definition &mlid, ml_uint index_of_feature_to_predict, const ml_instance &instance, 
					const ml_feature_value *result, ml_regression_results &mlrr);
void printRegressionResultsSummary(const ml_regression_results &mlrr);


bool writeModelJSONToFile(const ml_string &path_to_file, cJSON *json_object);
cJSON *readModelJSONFromFile(const ml_string &path_to_file);

extern const ml_string ML_UNKNOWN_DISCRETE_CATEGORY;

} // namespace puml




