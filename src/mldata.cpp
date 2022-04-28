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

#include "mldata.h"
#include "mlutil.h"
#include "rapidcsv.h"

#include <iostream>
#include <algorithm>
#include <limits>
#include <string>
#include <random>
#include <chrono>
#include <math.h>
#include <string.h>



namespace puml {

const ml_string ML_VERSION_STRING = "0.2";
const ml_float ML_VERSION = 1.0; 
const ml_string ML_UNKNOWN_DISCRETE_CATEGORY = "<unknown>";
const ml_float MISSING_CONTINUOUS_FEATURE_VALUE = std::numeric_limits<ml_float>::lowest();
const ml_uint ML_DEFAULT_SEED = 999;

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

  rapidcsv::Document doc(path_to_input_file, rapidcsv::LabelParams(-1, -1)); 
  int row_count = doc.GetRowCount();
  if(row_count <= 0) {
    log_error("can't open input file %s\n", path_to_input_file.c_str());
    return(false);
  }

  ml_vector<ml_stats_helper> stats_helper;
  ml_map<ml_uint, bool> ignored_features;

  for(int row_idx = 0; row_idx < row_count; ++row_idx) {

    ml_vector<ml_string> features_as_string = doc.GetRow<ml_string>(row_idx);
    if(features_as_string.empty() || (features_as_string.size() == 1)) { // empty line
      continue;
    }

    if(stats_helper.size() == 0) {
      //
      // The first line defines each feature. For example, Feature1:C,Feature2:C,Feature3:D,Feature4:I,... 
      // defines Feature1 and Feature2 as a continuous features, Feature3 as discrete, and Feature4 is ignored. 
      //
      ml_instance_definition mlid_temp;
      if(!initInstanceDefinition(mlid_temp, features_as_string, stats_helper, ignored_features)) {
	log_error("confused by instance definition line:%d\n", row_idx);
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
	log_error("confused by instance row:%d\n", row_idx);
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


static void fillJSONObjectFromInstanceDefinition(json &json_mlid, const ml_instance_definition &mlid) {

  json_mlid["object"] = "ml_instance_definition";
  json_mlid["version"] = ML_VERSION_STRING;
  
  json json_fdescs = json::array();
  for(std::size_t ii =0; ii < mlid.size(); ++ii) {
    json json_feature_desc;
    json_feature_desc["name"] = mlid[ii]->name;
    json_feature_desc["type"] = mlid[ii]->type;
    json_feature_desc["missing"] = mlid[ii]->missing;
    json_feature_desc["preserve_missing"] = mlid[ii]->preserve_missing;
    
    if(mlid[ii]->type == ml_feature_type::continuous) {
      json_feature_desc["mean"] = mlid[ii]->mean;
      json_feature_desc["sd"] = mlid[ii]->sd;
    }
    else {
      json_feature_desc["discrete_mode_index"] = mlid[ii]->discrete_mode_index;

      json json_values_array = json::array();
      for(std::size_t jj = 0; jj < mlid[ii]->discrete_values.size(); ++jj) {
	json_values_array.push_back(mlid[ii]->discrete_values[jj]);
      }
      json_feature_desc["discrete_values"] = json_values_array;
      
      json json_values_count_array = json::array();
      for(std::size_t jj = 0; jj < mlid[ii]->discrete_values_count.size(); ++jj) {
	json_values_count_array.push_back(mlid[ii]->discrete_values_count[jj]);
      }
      json_feature_desc["discrete_values_count"] = json_values_count_array;
      
    }

    json_fdescs.push_back(json_feature_desc);
  }

  json_mlid["fdesc_array"] = json_fdescs;
  
}


static bool createInstanceDefinitionFromJSONObject(const json &json_object, ml_instance_definition &mlid) {

  if(json_object.empty()) {
    return(false);
  }
  
  ml_string object_name = json_object["object"];
  if(object_name != "ml_instance_definition") {
    log_error("json object is not an instance definition...\n");
    return(false);
  }

  const json &fdesc_array = json_object["fdesc_array"];
  if(!fdesc_array.is_array() || (fdesc_array.size() == 0)) {
    log_error("json object is missing a fdsec array\n");
    return(false);
  }

  for(const json &fdesc : fdesc_array) {

    ml_feature_desc_ptr mlfd = std::make_shared<ml_feature_desc>();

    ml_uint fdesc_type=0, fdesc_missing=0;
    bool fdesc_preserve_missing=false;
    if(!fdesc.contains<ml_string>("name") ||
       !fdesc["name"].is_string() ||
       !get_numeric_value_from_json(fdesc, "type", fdesc_type) ||
       !get_numeric_value_from_json(fdesc, "missing", fdesc_missing) ||
       !get_bool_value_from_json(fdesc, "preserve_missing", fdesc_preserve_missing)) {
      log_error("malformed instance definition\n");
      return(false);
    }
				 
    mlfd->name = fdesc["name"];
    mlfd->type = (ml_feature_type) fdesc_type;
    mlfd->missing = fdesc_missing;
    mlfd->preserve_missing = fdesc_preserve_missing;
    
    if(mlfd->type == ml_feature_type::continuous) {
      ml_float fdesc_mean=0.0, fdesc_sd=0.0;
      if(!get_float_value_from_json(fdesc, "mean", fdesc_mean) ||
	 !get_float_value_from_json(fdesc, "sd", fdesc_sd)) {
	log_error("missing mean/sd from instance definition\n");
	return(false);
      }
      mlfd->mean = fdesc_mean;
      mlfd->sd = fdesc_sd;
    }
    else {
      ml_uint dmi=0;
      if(!get_numeric_value_from_json(fdesc, "discrete_mode_index", dmi)) {
	log_error("missing discrete feature index\n");
	return(false);
      }
      
      mlfd->discrete_mode_index = dmi;

      if(!fdesc.contains<ml_string>("discrete_values") ||
	 !fdesc["discrete_values"].is_array()) {
	log_error("missing discrete feature values\n");
	return(false);
      }
      
      const json &values_array = fdesc["discrete_values"];
      for(ml_uint jj=0; jj < values_array.size(); ++jj) {
	mlfd->discrete_values.push_back(values_array[jj]);
	mlfd->discrete_values_map[values_array[jj]] = jj;
      }

      if(!fdesc.contains<ml_string>("discrete_values_count") ||
	 !fdesc["discrete_values_count"].is_array()) {
	log_error("missing discrete feature values counts\n");
	return(false);
      }
      
      const json &values_count_array = fdesc["discrete_values_count"];
      for(ml_uint jj=0; jj < values_count_array.size(); ++jj) {
	mlfd->discrete_values_count.push_back(values_count_array[jj]);
      }
    }
    
    mlid.push_back(mlfd);
  }

  return(true);
}


bool write_instance_definition_to_file(const ml_string &path_to_file, 
				       const ml_instance_definition &mlid) {
  json json_mlid;
  fillJSONObjectFromInstanceDefinition(json_mlid, mlid);
  std::ofstream mlidout(path_to_file);
  mlidout << std::setw(4) << json_mlid << std::endl; 

  return(true);
}


bool read_instance_definition_from_file(const ml_string &path_to_file, ml_instance_definition &mlid) {
  mlid.clear();
  std::ifstream jsonfile(path_to_file);
  json json_mlid;
  jsonfile >> json_mlid;

  bool status = createInstanceDefinitionFromJSONObject(json_mlid, mlid);

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
