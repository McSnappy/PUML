# PUML
Poor Unwashed Machine Learner
-----------------------------

A simple-headed C++11 implementation of Decision Trees / Random Forest

The included sample program, mltest, uses the [Iris](https://archive.ics.uci.edu/ml/datasets/Iris "") and [Covertype](https://archive.ics.uci.edu/ml/datasets/Covertype "") datasets from UCI to demonstrate building trees and forests.  

PUML uses [rapidcsv](https://github.com/d99kris/rapidcsv "rapidcsv") (BSD-3) to handle csv data
and [this](https://github.com/nlohmann/json "json") (MIT) for json
  
  
Build and run mltest:
---------------------
    cd src  
    make  
    ./mltest  



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


