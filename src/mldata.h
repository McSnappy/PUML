/*
Copyright (c) Carl Sherrell

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

#include <memory>
#include <random>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "logging.h"

#include "json.hpp"
using json = nlohmann::json;

namespace puml {

// 
// ml_float is used to represent continuous feature values.
// ml_double is used when we operate on continuous feature values.
// see ml_feature_value below
//
using ml_float = float;
using ml_double = double;
using ml_uint = uint32_t;
using ml_string = std::string;


enum class ml_feature_type : ml_uint {
  continuous = 0,
  discrete // categorical
};


enum class ml_model_type : ml_uint {
  classification = 0,
  regression
};

extern const ml_string ML_VERSION_STRING;
extern const ml_float ML_VERSION; 
extern const ml_float MISSING_CONTINUOUS_FEATURE_VALUE;
extern const ml_uint ML_DEFAULT_SEED;

template <typename T>
using ml_vector = std::vector<T>;

template <typename T, typename T2>
using ml_map = std::unordered_map<T, T2>;

template <typename T>
using ml_set = std::unordered_set<T>;


//
// Random Number Generator
//
using ml_rng_engine = std::mt19937;

class ml_rng {
  public:
  ml_rng(ml_uint seed=ML_DEFAULT_SEED) : rng_(seed) {}
  ml_uint random_number() { return(rng_()); };
  
  private:
  ml_rng_engine rng_;
};


//
// we need our own shuffle since std::shuffle isn't guaranteed
// to give identical results across machines and compilers
//
template <typename T>
void shuffle_vector(ml_vector<T> &vec, ml_rng &rng);

template <typename T>
void shuffle_vector(ml_vector<T> &vec, ml_rng &rng) {
  for (std::size_t ii = vec.size() - 1; ii > 0; --ii) {
    std::swap(vec[ii], vec[rng.random_number() % (ii + 1)]);
  }
}


//
// ml_feature_desc holds info about a feature's name, type, distribution, etc
//
struct ml_feature_desc {

  ml_feature_type type;
  ml_string name;
  ml_uint missing = 0;
  
  // false (default): use feature's global mean/mode. 
  // true: insert an out of range value for continuous features 
  // and use a separate category for discrete. This option is 
  // available from the instance defintion row, the first row, 
  // of the data set. see load_data() below
  bool preserve_missing = false; 

  // continuous features
  ml_float mean = 0.0;
  ml_float sd = 0.0;

  // discrete features
  ml_vector<ml_string> discrete_values;
  ml_map<ml_string, ml_uint> discrete_values_map;
  ml_vector<ml_uint> discrete_values_count;
  ml_uint discrete_mode_index = 0;

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
using ml_feature_desc_ptr = std::shared_ptr<ml_feature_desc>;
using ml_instance_definition = ml_vector<ml_feature_desc_ptr>;


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
// load_data(...)
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
// and a separate category for missing discrete features will be used. The 
// default will use the feature's global mean or mode to populate missing values.
//
bool load_data(const ml_string &path_to_input_file, ml_instance_definition &mlid, ml_data &mld);


//
// The internal category values for discrete features are defined based on the order that each
// distinct category is found in the data when loaded with load_data(). If you
// wish to run a model built from training data on test data from a separate file you should
// use load_data_using_instance_definition() to force the load of test data 
// to use category definitions from the training data's ml_instance_definition.
// 
// returns true on success
//
// ids is an optional parameter that will be populated with the first column of the data file (id column
// of test data from kaggle competition, etc).
//
bool load_data_using_instance_definition(const ml_string &path_to_input_file, const ml_instance_definition &mlid, 
					 ml_data &mld, ml_vector<ml_string> *ids = nullptr);


//
// populate training and test vectors with instances randomly chosen from mld.
// after the call, training will have training_factor fraction of the data, test will have 
// the remainder and mld will be empty.
//
void split_data_into_training_and_test(ml_data &mld, ml_float training_factor, 
				       ml_data &training, ml_data &test,
				       ml_uint seed=ML_DEFAULT_SEED);


//
// Displays a summary of features including name, type, distribution, etc.
//
void print_data_summary(const ml_instance_definition &mlid);


//
// Returns the internal column index for a feature with the given name
//
ml_uint index_of_feature_with_name(const ml_string &feature_name, const ml_instance_definition &mlid);


//
// One Hot Encoding: Converts discrete (categorical) features to continuous binary features
//
bool create_onehotencoding_for_data(const ml_instance_definition &mlid, const ml_data &mld, 
				    const ml_string &name_of_feature_to_predict, 
				    ml_instance_definition &mlid_ohe, ml_data &mld_ohe);


//
// Save/Read Instance Definition To Disk (JSON)
//
bool write_instance_definition_to_file(const ml_string &path_to_file, 
				       const ml_instance_definition &mlid);
bool read_instance_definition_from_file(const ml_string &path_to_file, 
					ml_instance_definition &mlid);

extern const ml_string ML_UNKNOWN_DISCRETE_CATEGORY;

} // namespace puml
