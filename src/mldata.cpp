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

#include "mldata.h"

#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <string>
#include <random>
#include <chrono>
#include <math.h>
#include <string.h>

namespace puml {

const ml_float ML_VERSION = 0.1; 
const ml_string ML_UNKNOWN_DISCRETE_CATEGORY = "<unknown>";
const ml_float MISSING_CONTINUOUS_FEATURE_VALUE = std::numeric_limits<ml_float>::lowest();
const ml_uint ML_DEFAULT_SEED = 42;

//
// one per feature, used for online avg/variance calc 
// and to track which instances have missing data
//
typedef struct {
  ml_uint count;
  ml_double mean;
  ml_double M2;

  ml_vector<ml_uint> missing_data_instance_indices;

} ml_stats_helper;


static ml_string stringTrimLeadingTrailingWhitespace(ml_string &str) {
  size_t first = str.find_first_not_of(' ');
  if(first == ml_string::npos) {
    return("");
  }

  size_t last = str.find_last_not_of(' ');
  return(str.substr(first, (last-first+1)));
}


static bool parseInstanceDataLine(const ml_string &instance_line, ml_vector<ml_string> &features_as_string) {

  const char ML_DATA_DELIM = ',';

  features_as_string.clear();

  if(instance_line == "") {
    return(false);
  }
  
  std::stringstream ss(instance_line);

  while(ss.good()) {
    ml_string feature_as_string;
    std::getline(ss, feature_as_string, ML_DATA_DELIM);
    feature_as_string = stringTrimLeadingTrailingWhitespace(feature_as_string);
    features_as_string.push_back(feature_as_string);
  }
  
  return(features_as_string.size() > 0);
}

static bool initInstanceDefinition(ml_instance_definition &mlid, const ml_vector<ml_string> &features_as_string, 
				   ml_vector<ml_stats_helper> &stats_helper, ml_map<ml_uint, bool> &ignored_features) {

  //
  // Parse each feature definition, which is expected to be of the form:
  // Name:Type:Optional, for example Feature1:C for a continuous feature, or
  // SomeFeature:D for a discrete/categorical feature, or Feature:I to ignore.
  // 
  // You can specify Feature1:C:P or Feature1:D:P to preserve any missing values.
  // With :P, an out of range value will be used for missing continuous features,
  // and a separate category for missing discrete features.  The default will
  // use the feature's global mean or mode to populate missing values.
  //

  mlid.clear();
  
  if(features_as_string.size() < 2) {
    log_error("instance needs at least 2 features...\n");
    return(false);
  }

  const char ML_INSTANCE_DEF_DELIM = ':';
  const ml_string &ML_INSTANCE_DEF_CONTINUOUS_TOKEN = "C";
  const ml_string &ML_INSTANCE_DEF_DISCRETE_TOKEN = "D";
  const ml_string &ML_INSTANCE_DEF_IGNORE_TOKEN = "I";
  const ml_string &ML_INSTANCE_DEF_PRESERVE_MISSING_TOKEN = "P";

  for(std::size_t ii=0; ii < features_as_string.size(); ++ii) {

    ml_vector<ml_string> instance_definition_strings;
    std::stringstream ss(features_as_string[ii]);

    while(ss.good()) {
      ml_string definition;
      std::getline(ss, definition, ML_INSTANCE_DEF_DELIM);
      definition = stringTrimLeadingTrailingWhitespace(definition);
      instance_definition_strings.push_back(definition);
    } 

    std::size_t def_strings_size = instance_definition_strings.size();
    if((def_strings_size < 2) || // name and type
       (def_strings_size > 3) || // could have an optional (preserve missing)
       ((instance_definition_strings[1] != ML_INSTANCE_DEF_CONTINUOUS_TOKEN) && 
	(instance_definition_strings[1] != ML_INSTANCE_DEF_DISCRETE_TOKEN) &&
	(instance_definition_strings[1] != ML_INSTANCE_DEF_IGNORE_TOKEN))) {
      log_error("expected Name:C/D/I or Name:C/D:P at column %zu of instance definition line. got '%s'\n", ii, features_as_string[ii].c_str());
      mlid.clear();
      return(false);
    }

    if(instance_definition_strings[1] == ML_INSTANCE_DEF_IGNORE_TOKEN) {
      ignored_features[ii] = true;
      continue;
    }

    ml_feature_desc_ptr mlfd = std::make_shared<ml_feature_desc>();
    mlfd->name = instance_definition_strings[0];
    mlfd->type = (instance_definition_strings[1] == ML_INSTANCE_DEF_CONTINUOUS_TOKEN) ? ml_feature_type::continuous : ml_feature_type::discrete;
    mlfd->preserve_missing = ((def_strings_size == 3) && (instance_definition_strings[2] == ML_INSTANCE_DEF_PRESERVE_MISSING_TOKEN)) ? true : false;
    mlid.push_back(mlfd);

    ml_stats_helper sh = {};
    stats_helper.push_back(sh);
  }

  return(true);
}

static bool isValueMissing(const ml_string &value) {
  return((value == "") || (value == "?") || (value == "NA"));
}

static void addDiscreteValueToFeatureDesc(const ml_string &value, int index, ml_feature_desc &mlfd) {
  mlfd.discrete_values.push_back(value);
  mlfd.discrete_values_map[value] = index;
  mlfd.discrete_values_count.push_back(0);
}

static ml_uint findDiscreteValueIndexForValue(const ml_string &value, ml_feature_desc &mlfd) {

  // Index 0 always represents the unknown/unavailable category
  if(mlfd.discrete_values.size() == 0) {
    addDiscreteValueToFeatureDesc(ML_UNKNOWN_DISCRETE_CATEGORY, 0, mlfd);
  }

  ml_uint index = 0;
  ml_string dcat = isValueMissing(value) ? ML_UNKNOWN_DISCRETE_CATEGORY : value;
  ml_map<ml_string, ml_uint>::iterator it = mlfd.discrete_values_map.find(dcat);
  if(it != mlfd.discrete_values_map.end()) {
    index = it->second;
  }
  else {
    index = mlfd.discrete_values.size();
    addDiscreteValueToFeatureDesc(dcat, index, mlfd);
  }
  
  return(index);
}
 
static void updateStatsHelperWithFeatureValue(ml_stats_helper &stats_helper, const ml_feature_value &mlfv) {
  stats_helper.count += 1;
  ml_double delta = mlfv.continuous_value - stats_helper.mean;
  stats_helper.mean = stats_helper.mean + (delta / stats_helper.count);
  stats_helper.M2 = stats_helper.M2 + (delta * (mlfv.continuous_value - stats_helper.mean));
}

static bool processInstanceFeatures(ml_instance_definition &mlid, ml_data &mld, const ml_vector<ml_string> &features_as_string, 
				    ml_vector<ml_stats_helper> &stats_helper, const ml_map<ml_uint, bool> &ignored_features) {  

  if(features_as_string.size() != (mlid.size() + ignored_features.size())) {
    log_error("feature count mismatch b/t data row (%zu) and instance definition row (%zu); ignored (%zu)\n", 
		 features_as_string.size(), mlid.size() + ignored_features.size(), ignored_features.size());
    return(false);
  }    

  ml_instance_ptr mli = std::make_shared<ml_instance>();
  ml_uint feature_index = 0;

  mli->reserve(mlid.size());

  for(std::size_t str_index=0; str_index < features_as_string.size(); ++str_index) {

    if(ignored_features.find(str_index) != ignored_features.end()) {
      continue;
    }

    ml_feature_value mlfv;
    if(isValueMissing(features_as_string[str_index])) {
      //
      // This instance is missing the value for this feature. We record the 
      // instance index so that later we can populate using the mean or mode.
      //
      stats_helper[feature_index].missing_data_instance_indices.push_back(mld.size());
      mlid[feature_index]->missing += 1;
    }
    else if(mlid[feature_index]->type == ml_feature_type::continuous) {
      //
      // Convert from string and update the online mean/variance calculation.
      //
      try {
	mlfv.continuous_value = stof(features_as_string[str_index]);
      }
      catch(...) {
	puml::log_error("non-numeric value: '%s' given for continuous feature '%s'\n",
		       features_as_string[str_index].c_str(), mlid[feature_index]->name.c_str());
	return(false);
      }

      updateStatsHelperWithFeatureValue(stats_helper[feature_index], mlfv);
    }
    else { 
      //
      // ml_feature_type::discrete 
      //
      mlfv.discrete_value_index = findDiscreteValueIndexForValue(features_as_string[str_index], *mlid[feature_index]);
      mlid[feature_index]->discrete_values_count[mlfv.discrete_value_index] += 1;
    }

    mli->push_back(mlfv);
    ++feature_index;
  }
    
  mld.push_back(mli);

  return(true);
}

static void findModeValueIndexForDiscreteFeature(ml_feature_desc &mlfd) {
  
  if(mlfd.type != ml_feature_type::discrete) {
    return;
  }

  ml_uint mindex=0, mmax=0;
  for(std::size_t ii=1; ii < mlfd.discrete_values_count.size(); ++ii) {
    if(mlfd.discrete_values_count[ii] > mmax) {
      mmax = mlfd.discrete_values_count[ii];
      mindex = ii;
    }
  }

  mlfd.discrete_mode_index = mindex;
}

static void calcMeanOrModeOfFeatures(ml_instance_definition &mlid, ml_data &mld, ml_vector<ml_stats_helper> &stats_helper) {
  //
  // iterate over all features and compute the mean/std for continuous, and find the mode for discrete features.
  //
  for(std::size_t ii = 0; ii < mlid.size(); ++ii) {
    switch(mlid[ii]->type) {
    case ml_feature_type::continuous:
      mlid[ii]->mean = stats_helper[ii].mean;
      mlid[ii]->sd = (stats_helper[ii].count < 2) ? 0.0 : (sqrt(stats_helper[ii].M2 / (stats_helper[ii].count - 1)));
      break;
    case ml_feature_type::discrete:
      findModeValueIndexForDiscreteFeature(*mlid[ii]);
      break;  
    default: log_warn("unknown feature type...\n");
      break;
    }
  }
}

static void fillMissingInstanceFeatureValues(ml_instance_definition &mlid, ml_data &mld, ml_vector<ml_stats_helper> &stats_helper) {
  
  //
  // fill in mean or mode (unless preserving missing for that feature) for all instances with missing values
  //
  for(std::size_t findex=0; findex < stats_helper.size(); ++findex) {
    for(std::size_t jj=0; jj < stats_helper[findex].missing_data_instance_indices.size(); ++jj) {
      ml_uint instance_index = stats_helper[findex].missing_data_instance_indices[jj];
      ml_instance &instance = (*mld[instance_index]);
      if(mlid[findex]->type == ml_feature_type::continuous) {
	instance[findex].continuous_value = mlid[findex]->preserve_missing ? MISSING_CONTINUOUS_FEATURE_VALUE : mlid[findex]->mean;
	//log("setting missing cont feature %zu of instance %d to %.3f\n", findex, instance_index, instance[findex].continuous_value); 
      }
      else {
	instance[findex].discrete_value_index = mlid[findex]->preserve_missing ? 0 : mlid[findex]->discrete_mode_index;
	//log("setting missing cont feature %zu of instance %d to index %d\n", findex, instance_index, instance[findex].discrete_value_index);
      }
    }
  }

}

static bool instanceDefinitionsMatch(const ml_instance_definition &mlid, const ml_instance_definition &mlid_temp, bool discreteCategoryCheck) {
  if(mlid.size() != mlid_temp.size()) {
    return(false);
  }

  for(std::size_t ii = 0; ii < mlid.size(); ++ii) {
    if(mlid[ii]->type != mlid_temp[ii]->type) {
      return(false);
    }
    
    if(mlid[ii]->name != mlid_temp[ii]->name) {
      return(false);
    }

    if(discreteCategoryCheck && (mlid[ii]->type == ml_feature_type::discrete)) {
      if(mlid[ii]->discrete_values.size() < mlid_temp[ii]->discrete_values.size()) {
	log_warn("category count mismatch: %zu vs %zu, feature %s\n", mlid[ii]->discrete_values.size()-1, mlid_temp[ii]->discrete_values.size()-1, mlid[ii]->name.c_str());
      }
    }
  }

  return(true);
}

static bool instanceDefinitionsMatchInCountTypeAndName(const ml_instance_definition &mlid, const ml_instance_definition &mlid_temp) {
  return(instanceDefinitionsMatch(mlid, mlid_temp, false));
}

static bool loadInstanceDataFromFile(const ml_string &path_to_input_file, ml_instance_definition &mlid, ml_data &mld, ml_vector<ml_string> *ids) {

  mld.clear();
  bool mlid_preloaded = mlid.empty() ? false : true;

  std::ifstream input_file(path_to_input_file);
  if(!input_file) {
    log_error("can't open input file %s\n", path_to_input_file.c_str());
    return(false);
  }

  ml_vector<ml_stats_helper> stats_helper;
  ml_map<ml_uint, bool> ignored_features;

  ml_string line;
  while(std::getline(input_file, line)) {

    // 
    // Parse each column into a vector of strings
    //
    ml_vector<ml_string> features_as_string;
    if(!parseInstanceDataLine(line, features_as_string)) {
      continue;
    }

    if(stats_helper.size() == 0) {
      //
      // The first line defines each feature. For example, Feature1:C,Feature2:C,Feature3:D,Feature4:I,... 
      // defines Feature1 and Feature2 as a continuous features, Feature3 as discrete, and Feature4 is ignored. 
      //
      ml_instance_definition mlid_temp;
      if(!initInstanceDefinition(mlid_temp, features_as_string, stats_helper, ignored_features)) {
	log_error("confused by instance definition line:%s\n", line.c_str());
	return(false);
      }

      if(!mlid_preloaded) {
	mlid = mlid_temp;
      }
      else if(!instanceDefinitionsMatchInCountTypeAndName(mlid, mlid_temp)) {
	log_error("file format doesn't match preloaded instance definition\n");
	return(false);
      }

    }
    else {
      //
      // Update stats for each feature, add the instance to ml_data, etc
      //
      if(!processInstanceFeatures(mlid, mld, features_as_string, stats_helper, ignored_features)) {
	log_error("confused by instance row:%s\n", line.c_str());
	return(false);
      }

      // 
      // we store the first column (assumed to be the instance id) if a vector is given
      //
      if(ids) {
	ids->push_back(features_as_string[0]);
      }
    }

  }
  
  input_file.close();
  
  if(!mlid_preloaded) {
    calcMeanOrModeOfFeatures(mlid, mld, stats_helper);
  }

  fillMissingInstanceFeatureValues(mlid, mld, stats_helper);

  return(true);
}

bool load_data_using_instance_definition(const ml_string &path_to_input_file, 
					 const ml_instance_definition &mlid, 
					 ml_data &mld, ml_vector<ml_string> *ids) {

  ml_instance_definition temp_mlid(mlid);
  if(!loadInstanceDataFromFile(path_to_input_file, temp_mlid, mld, ids)) {
    return(false);
  }

  if(!instanceDefinitionsMatch(mlid, temp_mlid, true)) {
    mld.clear();
    log_error("file format doesn't match preloaded instance definition...\n");
  }

  return(true);
}

bool load_data(const ml_string &path_to_input_file, ml_instance_definition &mlid, ml_data &mld) {
  mlid.clear();
  return(loadInstanceDataFromFile(path_to_input_file, mlid, mld, nullptr));
}

void print_data_summary(const ml_instance_definition &mlid) {

  log("\n\n*** Data Summary ***\n\n");

  for(std::size_t ii = 0; ii < mlid.size(); ++ii) {
    log("feature %zu: %s, missing: %d\n", ii, mlid[ii]->name.c_str(), mlid[ii]->missing);
    if(mlid[ii]->type == ml_feature_type::continuous) {
      log("     mean: %.3f  std: %.3f\n\n", mlid[ii]->mean, mlid[ii]->sd);
    }
    else {
      const ml_feature_desc &mlfd = *mlid[ii];
      for(std::size_t jj=1; jj < mlfd.discrete_values.size(); ++jj) {
	log("  category %zu: %s, count: %d\n", jj, mlfd.discrete_values[jj].c_str(), mlfd.discrete_values_count[jj]);
      }
      log("\n");
    }
  }

}


void split_data_into_training_and_test(ml_data &mld, ml_float training_factor, 
				       ml_data &training, ml_data &test,
				       ml_uint seed) {
  
  training.clear();
  test.clear();

  if(mld.empty()) {
    return;
  }
 
  if(training_factor > 0.99) {
    log_error("bogus training factor %.2f\n", training_factor);
    return;
  }

  ml_rng rng(seed);
  shuffle_vector(mld, rng);
  ml_uint training_size = (ml_uint) ((training_factor * mld.size()) + 0.5);
  training = ml_data(mld.begin(), mld.begin() + training_size);
  mld.erase(mld.begin(), mld.begin() + training_size);
  test = ml_data(mld.begin(), mld.end());
}

bool write_model_json_to_file(const ml_string &path_to_file, 
			      cJSON *json_object) {

  if(!json_object) {
    log_error("nil json object...\n");
    return(false);
  }

  cJSON_AddNumberToObject(json_object, "version", ML_VERSION);
  
  //char *json_str = cJSON_Print(json_object);
  char *json_str = cJSON_PrintUnformatted(json_object);
  if(!json_str) {
    log_error("failed to convert json object to string...\n");
    return(false);
  }

  FILE *fp = fopen(path_to_file.c_str(), "w");
  if(!fp) {
    log_error("couldn't create model file: %s\n", path_to_file.c_str());
    free(json_str);
    return(false);
  }

  fprintf(fp, "%s\n", json_str);
  fclose(fp);
  free(json_str);

  return(true);
}

cJSON *read_model_json_from_file(const ml_string &path_to_file) {
  FILE *fp = fopen(path_to_file.c_str(), "r");
  if(!fp) {
    log_error("couldn't open model file: %s\n", path_to_file.c_str());
    return(nullptr);
  }

  fseek(fp, 0, SEEK_END);
  long len = ftell(fp);
  fseek(fp,0,SEEK_SET);

  if(len == 0) {
    log_error("model file is empty: %s\n", path_to_file.c_str());
    return(nullptr);
  }

  char *json_str = (char *) malloc(len+1);
  if(!json_str) {
    log_error("out of memory...\n");
    return(nullptr);
  }

  if(!fread(json_str, 1, len, fp)) {
    log_error("failed to read model json: %s\n", path_to_file.c_str());
    return(nullptr);
  }

  fclose(fp);

  cJSON *json_obj = cJSON_Parse(json_str);
  if(!json_obj) {
    log_error("failed to parse model json: %s\n", path_to_file.c_str());
  }

  free(json_str);

  return(json_obj);
}

static cJSON *createJSONObjectFromInstanceDefinition(const ml_instance_definition &mlid) {

  cJSON *json_mlid = cJSON_CreateObject();
  cJSON_AddStringToObject(json_mlid, "object", "ml_instance_definition");

  cJSON *json_fdescs = cJSON_CreateArray();
  cJSON_AddItemToObject(json_mlid, "fdesc_array", json_fdescs);

  for(std::size_t ii =0; ii < mlid.size(); ++ii) {
    cJSON *json_feature_desc = cJSON_CreateObject();
    cJSON_AddStringToObject(json_feature_desc, "name", mlid[ii]->name.c_str());
    cJSON_AddNumberToObject(json_feature_desc, "type", (double) mlid[ii]->type);
    cJSON_AddNumberToObject(json_feature_desc, "missing", mlid[ii]->missing);
    cJSON_AddNumberToObject(json_feature_desc, "preserve_missing", (mlid[ii]->preserve_missing ? 1 : 0));
    
    if(mlid[ii]->type == ml_feature_type::continuous) {
      cJSON_AddNumberToObject(json_feature_desc, "mean", mlid[ii]->mean);
      cJSON_AddNumberToObject(json_feature_desc, "sd", mlid[ii]->sd);
    }
    else {
      cJSON_AddNumberToObject(json_feature_desc, "discrete_mode_index", mlid[ii]->discrete_mode_index);

      cJSON *json_values_array = cJSON_CreateArray();
      cJSON_AddItemToObject(json_feature_desc, "discrete_values", json_values_array);
      for(std::size_t jj = 0; jj < mlid[ii]->discrete_values.size(); ++jj) {
	cJSON_AddItemToArray(json_values_array, cJSON_CreateString(mlid[ii]->discrete_values[jj].c_str()));
      }
      
      cJSON *json_values_count_array = cJSON_CreateArray();
      cJSON_AddItemToObject(json_feature_desc, "discrete_values_count", json_values_count_array);
      for(std::size_t jj = 0; jj < mlid[ii]->discrete_values_count.size(); ++jj) {
	cJSON_AddItemToArray(json_values_count_array, cJSON_CreateNumber(mlid[ii]->discrete_values_count[jj]));
      }
    }

    cJSON_AddItemToArray(json_fdescs, json_feature_desc);
  }


  return(json_mlid);
}

static bool validateJSONFeatureDesc(cJSON *fdesc) {

  if(!cJSON_GetObjectItem(fdesc, "name") ||
     !cJSON_GetObjectItem(fdesc, "type") ||
     !cJSON_GetObjectItem(fdesc, "missing") ||
     !cJSON_GetObjectItem(fdesc, "preserve_missing")) {
    return(false);
  }

  ml_feature_type type = (ml_feature_type) cJSON_GetObjectItem(fdesc, "type")->valueint;
  if(type == ml_feature_type::continuous) {
    if(!cJSON_GetObjectItem(fdesc, "mean") ||
       !cJSON_GetObjectItem(fdesc, "sd")) {
      return(false);
    }
  }
  else {
    if(!cJSON_GetObjectItem(fdesc, "discrete_mode_index") ||
       !cJSON_GetObjectItem(fdesc, "discrete_values") ||
       !cJSON_GetObjectItem(fdesc, "discrete_values_count")) {
      return(false);
    }
  }

  return(true);
}

static bool createInstanceDefinitionFromJSONObject(cJSON *json_object, ml_instance_definition &mlid) {

  if(!json_object) {
    log_error("nil json object...\n");
    return(false);
  }

  cJSON *object = cJSON_GetObjectItem(json_object, "object");
  if(!object || !object->valuestring || strcmp(object->valuestring, "ml_instance_definition")) {
    log_error("json object is not an instance definition...\n");
    return(false);
  }

  cJSON *fdesc_array = cJSON_GetObjectItem(json_object, "fdesc_array");
  if(!fdesc_array || (fdesc_array->type != cJSON_Array)) {
    log_error("json object is missing a fdsec array\n");
    return(false);
  }

  int fdesc_count = cJSON_GetArraySize(fdesc_array);
  if(fdesc_count <= 0) {
    log_error("json object has empty fdesc array\n");
    return(false);
  }

  for(int ii=0; ii < fdesc_count; ++ii) {
    cJSON *fdesc = cJSON_GetArrayItem(fdesc_array, ii);
    if(!fdesc || (fdesc->type != cJSON_Object)) {
      log_error("json object has bogus fdesc in fdesc array\n");
      return(false);
    }

    if(!validateJSONFeatureDesc(fdesc)) {
      log_error("json object has missing/invalid fdesc attributes: index %d\n", ii);
      return(false);
    }

    ml_feature_desc_ptr mlfd = std::make_shared<ml_feature_desc>();
    mlfd->name = cJSON_GetObjectItem(fdesc, "name")->valuestring;
    mlfd->type = (ml_feature_type) cJSON_GetObjectItem(fdesc, "type")->valueint;
    mlfd->missing = cJSON_GetObjectItem(fdesc, "missing")->valueint;
    mlfd->preserve_missing = (cJSON_GetObjectItem(fdesc, "missing")->valueint > 0) ? true : false;
    
    if(mlfd->type == ml_feature_type::continuous) {
      mlfd->mean = cJSON_GetObjectItem(fdesc, "mean")->valuedouble;
      mlfd->sd = cJSON_GetObjectItem(fdesc, "sd")->valuedouble;
    }
    else {
      mlfd->discrete_mode_index = cJSON_GetObjectItem(fdesc, "discrete_mode_index")->valueint;
      
      cJSON *values_array = cJSON_GetObjectItem(fdesc, "discrete_values");
      for(int jj=0; jj < cJSON_GetArraySize(values_array); ++jj) {
	mlfd->discrete_values.push_back(cJSON_GetArrayItem(values_array, jj)->valuestring);
	mlfd->discrete_values_map[cJSON_GetArrayItem(values_array, jj)->valuestring] = jj;
      }

      cJSON *values_count_array = cJSON_GetObjectItem(fdesc, "discrete_values_count");
      for(int jj=0; jj < cJSON_GetArraySize(values_count_array); ++jj) {
	mlfd->discrete_values_count.push_back(cJSON_GetArrayItem(values_count_array, jj)->valueint);
      }
    }
    
    mlid.push_back(mlfd);
  }

  return(true);
}


bool write_instance_definition_to_file(const ml_string &path_to_file, 
				       const ml_instance_definition &mlid) {
  cJSON *json_object = createJSONObjectFromInstanceDefinition(mlid);
  if(!json_object) {
    log_error("couldn't create json object from instance definition\n");
    return(false);
  }

  bool status = write_model_json_to_file(path_to_file, json_object);
  cJSON_Delete(json_object);

  return(status);
}


bool read_instance_definition_from_file(const ml_string &path_to_file, ml_instance_definition &mlid) {
  mlid.clear();
  cJSON *json_object = read_model_json_from_file(path_to_file);
  if(!json_object) {
    log_error("couldn't load instance definition json object from model file: %s\n", path_to_file.c_str());
    return(false);
  }

  bool status = createInstanceDefinitionFromJSONObject(json_object, mlid);
  cJSON_Delete(json_object);

  return(status);
}


static void createOneHotEncodingInstanceDefinition(const ml_instance_definition &mlid, const ml_string &name_of_index_to_predict, 
						   ml_instance_definition &mlid_ohe, ml_vector<ml_stats_helper> &stats_helper) {
  //
  // create the new mlid with discrete feature categories mapped to continuous features
  //
  for(std::size_t findex = 0; findex < mlid.size(); ++findex) {
    const ml_feature_desc &fdesc = *mlid[findex];
    if((fdesc.type == ml_feature_type::continuous) || (fdesc.name == name_of_index_to_predict)) {
      mlid_ohe.push_back(mlid[findex]);
      ml_stats_helper sh = {};
      stats_helper.push_back(sh);
    }
    else {
      for(std::size_t value_index = (fdesc.preserve_missing ? 0 : 1); value_index < fdesc.discrete_values.size(); ++value_index) {
	ml_feature_desc_ptr ohe_fdesc = std::make_shared<ml_feature_desc>();
	ohe_fdesc->type = ml_feature_type::continuous;
	ohe_fdesc->name = fdesc.name + "_" + fdesc.discrete_values[value_index];
	mlid_ohe.push_back(ohe_fdesc);
	ml_stats_helper sh = {};
	stats_helper.push_back(sh);
      }
    }
  }
}

static void createOneHotEncodingForData(const ml_instance_definition &mlid, const ml_data &mld, 
					const ml_string &name_of_feature_to_predict,
					ml_data &mld_ohe, ml_vector<ml_stats_helper> &stats_helper) {
  //
  // convert the instances to the new one hot encoded format
  //
  for(std::size_t ii=0; ii < mld.size(); ++ii) {
    const ml_instance &inst = *mld[ii];
    ml_instance_ptr inst_ohe = std::make_shared<ml_instance>();
    
    for(std::size_t findex = 0; findex < mlid.size(); ++findex) {
      const ml_feature_desc &fdesc = *mlid[findex];
   
      // just copy the continuous features (and feature to predict)
      if((fdesc.type == ml_feature_type::continuous) || (fdesc.name == name_of_feature_to_predict)) {
	inst_ohe->push_back(inst[findex]);
      }
      else { // use 1.0 for the ohe feature that represents this instance's category of the original discrete feature, 0.0 otherwise.
	for(std::size_t value_index = (fdesc.preserve_missing ? 0 : 1); value_index < fdesc.discrete_values.size(); ++value_index) {
	  ml_feature_value fv_ohe = {};
	  fv_ohe.continuous_value = (inst[findex].discrete_value_index == value_index) ? 1.0 : 0.0;
	  updateStatsHelperWithFeatureValue(stats_helper[inst_ohe->size()], fv_ohe);
	  inst_ohe->push_back(fv_ohe);
	}
      }
    }

    mld_ohe.push_back(inst_ohe);
  }
}

static void updateStatsForOneHotEncoding(ml_instance_definition &mlid_ohe, const ml_vector<ml_stats_helper> &stats_helper) {
  //
  // update the new ml_instance_defintion with mean/sd for the encoded features
  //
  for(std::size_t findex = 0; findex < mlid_ohe.size(); ++findex) {
    if(stats_helper[findex].count == 0) {
      continue;
    }

    mlid_ohe[findex]->mean = stats_helper[findex].mean;
    mlid_ohe[findex]->sd = (stats_helper[findex].count < 2) ? 0.0 : (sqrt(stats_helper[findex].M2 / (stats_helper[findex].count - 1)));
  }

}

bool create_onehotencoding_for_data(const ml_instance_definition &mlid, const ml_data &mld, 
				    const ml_string &name_of_feature_to_predict, 
				    ml_instance_definition &mlid_ohe, ml_data &mld_ohe) {

  mlid_ohe.clear();
  mld_ohe.clear();

  ml_vector<ml_stats_helper> stats_helper;
  createOneHotEncodingInstanceDefinition(mlid, name_of_feature_to_predict, mlid_ohe, stats_helper);
  createOneHotEncodingForData(mlid, mld, name_of_feature_to_predict, mld_ohe, stats_helper);
  updateStatsForOneHotEncoding(mlid_ohe, stats_helper);

  return(true);
}

ml_uint index_of_feature_with_name(const ml_string &feature_name, const ml_instance_definition &mlid) {
  for(std::size_t ii = 0; ii < mlid.size(); ++ii) {
    if(mlid[ii]->name == feature_name) {
      return(ii);
    }
  }

  ml_string msg = "index_of_feature_with_name(): couldn't find feature with name " + feature_name;
  throw std::invalid_argument(msg);
}


} //namespace puml
