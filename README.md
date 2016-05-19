# PUML
Poor Unwashed Machine Learner
-----------------------------

A simple-headed C++ implementation of Decision Trees, Random Forest, and kNN for classification and regression. 

The included sample program, mltest, uses the [Iris](https://archive.ics.uci.edu/ml/datasets/Iris "Title")  and [Covertype](https://archive.ics.uci.edu/ml/datasets/Covertype "Title") datasets from UCI to demonstrate building trees and forests.  

PUML uses Dave Gamble's [cJSON](https://github.com/DaveGamble/cJSON "Title") to read/write its models.
  
  
Build and run mltest:
---------------------
    cd src  
    make  
    ./mltest  


Load Data:
----------
    //
    // ml_loadInstanceDataFromFile()
    //
    // path_to_input_file -- csv file where each row represents an instance, and the first row is in
    //                       the instance definition format below.
    // ml_instance_definition -- will be populated with the features defined by the first row
    // ml_data -- will be populated with instance data from the csv
    //
    // returns true on success, false otherwise
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
    // bool ml_loadInstanceDataFromFile(const ml_string &path_to_input_file, ml_instance_definition &mlid, ml_data &mld);
    //
    // ex:
    ml_data iris_mld;
    ml_instance_definition iris_mlid;
    ml_loadInstanceDataFromFile("./iris.csv", iris_mlid, iris_mld);
  

Build A Decision Tree:
----------------------

    dt_tree iris_tree;
    dt_build_config dtbc = {};
    dtbc.max_tree_depth = 6;
    dtbc.min_leaf_instances = 2;
    dtbc.index_of_feature_to_predict = ml_indexOfFeatureWithName("Class", iris_mlid);
    if(!dt_buildDecisionTree(iris_mlid, iris_training, dtbc, iris_tree)) {
      ml_log_error("failed to build tree...\n");
      exit(1);
    }

    dt_printDecisionTreeSummary(iris_mlid, iris_tree); 

With Output:

    *** Decision Tree Summary ***
    
    Feature To Predict: Class
    Type: classification, Leaves: 6, Size: 11
    
    PetalLength <= 2.3: Iris-setosa
    PetalLength > 2.3
    |  PetalWidth <= 1.75
    |  |  PetalLength <= 4.95: Iris-versicolor
    |  |  PetalLength > 4.95
    |  |  |  PetalWidth <= 1.55: Iris-virginica
    |  |  |  PetalWidth > 1.55: Iris-versicolor
    |  PetalWidth > 1.75
    |  |  PetalLength <= 4.85: Iris-versicolor
    |  |  PetalLength > 4.85: Iris-virginica

  
And Results On A Holdout Set:  
    
    Instances: 75
    Correctly Classified: 72 (96%)
    
          a      b      c  <-- classified as
         30      0      0 | a = Iris-setosa
          0     20      0 | b = Iris-versicolor
          0      3     22 | c = Iris-virginica


Build A Random Forest:
----------------------
    rf_forest cover_forest;
    rf_build_config rfbc = {};
    rfbc.index_of_feature_to_predict = ml_indexOfFeatureWithName("CoverType", cover_mlid);
    rfbc.number_of_trees = 50;
    rfbc.number_of_threads = 2;
    rfbc.max_tree_depth = 30;
    rfbc.seed = 999;
    rfbc.max_continuous_feature_splits = 20; // experimental optimization
    rfbc.features_to_consider_per_node = (ml_uint)(sqrt(cover_mlid.size()-1) + 0.5);
    
    if(!rf_buildRandomForest(cover_mlid, cover_training, rfbc, cover_forest)) {
      ml_log_error("failed to build random forest...\n");
      exit(1);
    }

With Out Of Bag Error Output And Feature Importance (truncated):

    Instances: 58101
    Correctly Classified: 46302 (80%)
    
          a      b      c      d      e      f      g  <-- classified as
        127    721     40      0     38      2      0 | a = 5
          8  24691   3215     12    303     26      0 | b = 2
          1   4635  16439    124      6      4      0 | c = 1
          0     13    721   1373      0      0      0 | d = 7
          1    341      0      0   3223     50      4 | e = 3
          0    481      2      0    857    376      3 | f = 6
          0      0      0      0    189      2     73 | g = 4

      1.45 ST29 (631 nodes, 0.05)
      1.66 ST12 (1415 nodes, 0.03)
      1.73 ST9 (1841 nodes, 0.02)
      1.80 WA2 (9877 nodes, 0.00)
      2.11 ST23 (988 nodes, 0.05)
      2.40 ST28 (837 nodes, 0.07)
      2.60 ST22 (1059 nodes, 0.06)
      2.69 ST30 (1014 nodes, 0.06)
      3.11 WA1 (11730 nodes, 0.01)
      3.35 ST31 (1206 nodes, 0.06)
      3.50 WA3 (9058 nodes, 0.01)
      3.69 ST32 (1274 nodes, 0.07)
     48.51 Hillshade3pm (13012 nodes, 0.09)
     49.80 HillshadeNoon (14012 nodes, 0.08)
     52.91 Hillshade9am (14729 nodes, 0.08)
     59.25 Slope (17947 nodes, 0.08)
     64.03 VerticalDistance (17042 nodes, 0.09)
     67.09 HorizontalDistance (17894 nodes, 0.09)
     68.32 DistanceFire (14651 nodes, 0.11)
     74.99 Aspect (20939 nodes, 0.08)
     81.45 DistanceRoadways (18484 nodes, 0.10)
     100.00 Elevation (24614 nodes, 0.09)

Save A Random Forest:
---------------------
    rf_writeRandomForestToDirectory("./rf-cover", cover_mlid, cover_forest);

    The directory ./rf-cover will be created and filled with your trees:
    $ ls
    mlid.json		tree16.1463692228.json	tree24.1463692228.json	tree32.1463692228.json	tree40.1463692228.json	tree49.1463692228.json
    rf.json			tree17.1463692228.json	tree25.1463692228.json	tree33.1463692228.json	tree41.1463692228.json	tree5.1463692228.json
    tree1.1463692228.json	tree18.1463692228.json	tree26.1463692228.json	tree34.1463692228.json	tree42.1463692228.json	tree50.1463692228.json
    tree10.1463692228.json	tree19.1463692228.json	tree27.1463692228.json	tree35.1463692228.json	tree43.1463692228.json	tree6.1463692228.json
    tree11.1463692228.json	tree2.1463692228.json	tree28.1463692228.json	tree36.1463692228.json	tree44.1463692228.json	tree7.1463692228.json
    tree12.1463692228.json	tree20.1463692228.json	tree29.1463692228.json	tree37.1463692228.json	tree45.1463692228.json	tree8.1463692228.json
    tree13.1463692228.json	tree21.1463692228.json	tree3.1463692228.json	tree38.1463692228.json	tree46.1463692228.json	tree9.1463692228.json
    tree14.1463692228.json	tree22.1463692228.json	tree30.1463692228.json	tree39.1463692228.json	tree47.1463692228.json
    tree15.1463692228.json	tree23.1463692228.json	tree31.1463692228.json	tree4.1463692228.json	tree48.1463692228.json
    
    // use rf_readRandomForestFromDirectory() to load the forest. See the note 
    // on ml_loadInstanceDataFromFileUsingInstanceDefinition() 
    // for details on categorical data mapping when loading new test datasets

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


