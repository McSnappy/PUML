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

#include <iostream>

#include "mlmodel.h"
#include "decisiontree.h"
#include "randomforest.h"


void decision_tree_example();
void random_forest_example();

int main(int argc, char **argv) {

  decision_tree_example();
  random_forest_example();
 
  return 0;
}


void decision_tree_example() {

  std::cout << "+++ decision tree demo using iris data +++" << std::endl;

  // Load the Iris data
  puml::ml_data mld;
  puml::ml_instance_definition mlid;
  puml::load_data("./iris.csv", mlid, mld);

  // Take 50% for training
  puml::ml_data training, test;
  puml::split_data_into_training_and_test(mld, 0.5, training, test, 999);

  // Build a single decision tree with 
  // max depth of 6 and a minimum of 2 instances at 
  // leaf nodes.
  puml::decision_tree dt{mlid, "Class", 6, 2};
  dt.train(training);
  
  // Show the tree structure
  std::cout << dt.summary() << std::endl;

  // Test the tree using the holdout
  puml::ml_classification_results test_results{mlid, dt.index_of_feature_to_predict()};
  for(const auto &inst_ptr : test) {
    test_results.collect_result(dt.evaluate(*inst_ptr), *inst_ptr);
  }
  
  std::cout << "*** Holdout Results ***" << std::endl << test_results.summary();

}


void random_forest_example() {

  std::cout << "+++ random forest demo using cover type data +++" << std::endl;

  // Load the cover type data
  puml::ml_data mld;
  puml::ml_instance_definition mlid;
  puml::load_data("./covertype.csv", mlid, mld);
  
  // Take 10% for training (just for demonstration)
  puml::ml_data training,test;
  puml::split_data_into_training_and_test(mld, 0.1, training, test);

  // 3 fold cross validation, 50 trees per forest (for demonstration)
  puml::ml_model<puml::random_forest> rf{mlid, "CoverType", 50};
  auto cv = rf.train<puml::ml_classification_results>(training, 3, 333);
  std::cout <<  rf.model().feature_importance_summary() << std::endl;
  std::cout << cv.summary() << std::endl;
  std::cout << "testing using holdout..." << std::endl;
  auto test_results = rf.evaluate<puml::ml_classification_results>(test);
  std::cout << "*** Holdout Results ***" << std::endl << test_results.summary();

}


