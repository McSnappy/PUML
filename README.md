# PUML
Poor Unwashed Machine Learner
-----------------------------

A simple-headed C++11 implementation of Decision Trees, Random Forest, Boosted Trees, kNN, and k-means.

The included sample program, mltest, uses the [Iris](https://archive.ics.uci.edu/ml/datasets/Iris ""), [Covertype](https://archive.ics.uci.edu/ml/datasets/Covertype ""), and [Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality "") datasets from UCI to demonstrate building trees and forests.  

PUML uses Dave Gamble's [cJSON](https://github.com/DaveGamble/cJSON "cJSON") (MIT) to read/write its models, and John Burkardt's implementation of Brent's Method [BRENT](https://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.html "BRENT") (LGPL).
  
  
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


