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

#include <iostream>

#include "mlmodel.h"
#include "decisiontree.h"
#include "randomforest.h"
#include "boosting.h"


void decision_tree_example();
void random_forest_example();
void boosted_trees_example();

int main(int argc, char **argv) {

  decision_tree_example();
  random_forest_example();
  boosted_trees_example();
 
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
  
  // Take 10% for training (for simplicity)
  puml::ml_data training,test;
  puml::split_data_into_training_and_test(mld, 0.1, training, test);

  // 2 fold cross validation, 50 trees per forest (for simplicity)
  puml::ml_model<puml::random_forest> rf{mlid, "CoverType", 50};
  auto cv = rf.train<puml::ml_classification_results>(training, 2, 333);
  std::cout <<  rf.model().feature_importance_summary() << std::endl;
  std::cout << cv.summary() << std::endl;
  std::cout << "testing using holdout..." << std::endl;
  auto test_results = rf.evaluate<puml::ml_classification_results>(test);
  std::cout << "*** Holdout Results ***" << std::endl << test_results.summary();
}


void boosted_trees_example() {

  std::cout << "+++ boosted trees demo using wine quality data +++" << std::endl;

  // Load the Iris data
  puml::ml_data mld;
  puml::ml_instance_definition mlid;
  puml::load_data("./winequality-white.csv", mlid, mld);

  // Take 50% for training
  puml::ml_data training, test;
  puml::split_data_into_training_and_test(mld, 0.5, training, test, 222);

  // 100 trees in the ensemble, 0.1 learning rate, random seed of 111,
  // max depth of 8, subsample of 0.9
  puml::ml_model<puml::boosted_trees> bt{mlid, "quality", 100, 0.1, 111, 8, 0.9};

  // Custom Loss
  bt.model().set_loss_func([](puml::ml_double yi, puml::ml_double yhat) { 
      return(fabs(yi - yhat)); 
    });

  bt.model().set_gradient_func([](puml::ml_double yi, puml::ml_double yhat) { 
      puml::ml_double diff = yi - yhat;
      if(diff < 0.0) {
	diff = -1.0;
      }
      else if(diff > 0.0) {
	diff = 1.0;
      }
      
      return(diff);
    });

  // Progress while training
  bt.model().set_progress_callback([&bt, &test](puml::ml_uint iteration) -> bool {
      if((iteration % 10) == 0) {
	auto test_results = bt.evaluate<puml::ml_regression_results>(test);
	std::cout << std::endl << "*** Holdout Results at iteration " << iteration << " ***" << std::endl;
	std::cout << test_results.summary();
      }

      return(true);
    });

  // Train using 5 fold cross validation
  auto cv = bt.train<puml::ml_regression_results>(training, 5, 333);
  std::cout << cv.summary();
}
