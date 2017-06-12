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

#include "mlutil.h"
#include "decisiontree.h"

#include <ctime>
#include <dirent.h>
#include <sstream>
#include <sys/stat.h>

namespace puml {

const ml_string &TREE_MODEL_FILE_PREFIX = "tree";
  
bool prepare_directory_for_model_save(const ml_string &path_to_dir) {
  
  if((path_to_dir == ".") || (path_to_dir == "..")) {
    return(false);
  }
  
  //
  // move the directory if it already exists 
  //
  struct stat info;
  if((stat(path_to_dir.c_str(), &info) == 0) && (info.st_mode & S_IFDIR)) {
    
    std::time_t timestamp = std::time(0);    
    std::ostringstream ss;
    ss << "mv " << path_to_dir << " " << path_to_dir << "." << timestamp;
    if(system(ss.str().c_str())){
      log_error("couldn't replace previous model directory: %s\n", path_to_dir.c_str());
      return(false);
    }
  }

  //
  // create the model save directory 
  //
  if(mkdir(path_to_dir.c_str(), 0755)) {
    log_error("couldn't create model save directory: %s\n", path_to_dir.c_str());
    perror("ERROR --> mkdir");
    return(false);
  }
  
  
  return(true);
}
  
  
bool read_decision_trees_from_directory(const ml_string &path_to_dir,
					const ml_instance_definition &mlid,
					ml_vector<decision_tree> &trees) {   
  trees.clear();
  
  DIR *d = 0;
  struct dirent *dir = 0;
  d = opendir(path_to_dir.c_str());
  if(!d) {
    log_error("can't scan model directory: %s", path_to_dir.c_str());
    return(false);
  }
  
  while((dir = readdir(d)) != NULL) {
    ml_string file_name(dir->d_name);
    if(file_name.compare(0, TREE_MODEL_FILE_PREFIX.length(), 
			 TREE_MODEL_FILE_PREFIX) != 0) {
      continue;
    }
    
    ml_string full_path = path_to_dir + "/" + dir->d_name;
    decision_tree tree(full_path, mlid);
    trees.push_back(tree);
  }
  
  closedir(d);
  
  return(true);
}


static cJSON *get_number_item_from_json(cJSON *json_object, const ml_string &name) {
  cJSON *item = cJSON_GetObjectItem(json_object, name.c_str());
  if(!item || (item->type != cJSON_Number)) {
    log_error("json is missing %s\n", name.c_str());
    return(nullptr);
  }

  return(item);
}


bool get_numeric_value_from_json(cJSON *json_object, const ml_string &name, ml_uint &value) {
  cJSON *item = get_number_item_from_json(json_object, name);
  if(!item) {
    return(false);
  }
  value = item->valueint;
  return(true);
}


bool get_double_value_from_json(cJSON *json_object, const ml_string &name, ml_double &value) {
  cJSON *item = get_number_item_from_json(json_object, name);
  if(!item) {
    return(false);
  }
  value = item->valuedouble;
  return(true);
}


bool get_float_value_from_json(cJSON *json_object, const ml_string &name, ml_float &value) {
  ml_double dbl_value=0;
  get_double_value_from_json(json_object, name, dbl_value);
  value = dbl_value;
  return(true);
}


bool get_bool_value_from_json(cJSON *json_object, const ml_string &name, bool &value) {
  ml_uint value_as_int = 0;
  if(!get_numeric_value_from_json(json_object, name, value_as_int)) {
    return(false);
  }

  value = (value_as_int != 0) ? true : false;

  return(true);
}


bool get_modeltype_value_from_json(cJSON *json_object, const ml_string &name, ml_model_type &value) {
  ml_uint value_as_int = 0;
  if(!get_numeric_value_from_json(json_object, name, value_as_int)) {
    return(false);
  }

  value = (ml_model_type)value_as_int;

  return(true);
}

} // namespace puml
