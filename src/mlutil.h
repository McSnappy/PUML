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

  class decision_tree;

  //
  // Model Save/Restore Helpers
  //
  extern const ml_string &TREE_MODEL_FILE_PREFIX;

  bool prepare_directory_for_model_save(const ml_string &path_to_dir);

  bool read_decision_trees_from_directory(const ml_string &path_to_dir,
					  const ml_instance_definition &mlid,
					  ml_vector<decision_tree> &trees);

  bool get_numeric_value_from_json(cJSON *json_object, const ml_string &name, ml_uint &value);
  bool get_float_value_from_json(cJSON *json_object, const ml_string &name, ml_float &value);
  bool get_double_value_from_json(cJSON *json_object, const ml_string &name, ml_double &value);
  bool get_bool_value_from_json(cJSON *json_object, const ml_string &name, bool &value);
  bool get_modeltype_value_from_json(cJSON *json_object, const ml_string &name, ml_model_type &value);


  //
  // sprintf for string
  //
  template<typename ... Args>
  ml_string string_format(const ml_string &format, Args ... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return(ml_string(buf.get(), buf.get() + size - 1));
  }

}


