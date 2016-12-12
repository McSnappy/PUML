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
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>

namespace puml {

const ml_string &TREE_MODEL_FILE_PREFIX = "tree";
  
bool prepareDirectoryForModelSave(const ml_string &path_to_dir, 
				  bool overwrite_existing) {
  
  if((path_to_dir == ".") || (path_to_dir == "..")) {
    return(false);
  }
  
  // move the directory if it already exists and we are allowed to overwrite.
  struct stat info;
  if((stat(path_to_dir.c_str(), &info) == 0) && (info.st_mode & S_IFDIR)) {
    
    if(!overwrite_existing) {
      log_error("directory exists and we aren't allowed to overwrite\n");
      return(false);
    }

    std::time_t timestamp = std::time(0);    
    std::ostringstream ss;
    ss << "mv " << path_to_dir << " " << path_to_dir << "." << timestamp;
    if(system(ss.str().c_str())){
      log_error("couldn't replace previous model directory: %s\n", path_to_dir.c_str());
      return(false);
    }
  }

  // create the model save directory 
  if(mkdir(path_to_dir.c_str(), 0755)) {
    log_error("couldn't create model save directory: %s\n", path_to_dir.c_str());
    perror("ERROR --> mkdir");
    return(false);
  }
  
  
  return(true);
}
  
  
bool readDecisionTreesFromDirectory(const ml_string &path_to_dir,
				    ml_vector<dt_tree> &trees) {   
  trees.clear();
  
  DIR *d = 0;
  struct dirent *dir = 0;
  d = opendir(path_to_dir.c_str());
  if(!d) {
    return(false);
  }
  
  while((dir = readdir(d)) != NULL) {
    ml_string file_name(dir->d_name);
    if(file_name.compare(0, TREE_MODEL_FILE_PREFIX.length(), 
			 TREE_MODEL_FILE_PREFIX) != 0) {
      continue;
    }
    
    ml_string full_path = path_to_dir + "/" + dir->d_name;
    dt_tree tree = {};
    if(!readDecisionTreeFromFile(full_path, tree)) {
      log_error("failed to parse tree from json: %s\n", full_path.c_str());
      return(false);
    }
    
    trees.push_back(tree);
  }
  
  closedir(d);
  
  return(true);
}

} // namespace puml
