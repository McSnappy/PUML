
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "mldata.h"
#include "mlresults.h"

namespace puml {

template<typename T>
class ml_model final {
 public:
  template<typename... Args>
  ml_model(Args &&... args) : model_(std::forward<Args>(args)...) {}

  bool save(const ml_string &path) const { return(model_.save(path)); }
  bool restore(const ml_string &path) { return(model_.restore(path)); }
  
  template<typename U> 
  ml_crossvalidation_results<U> train(const ml_data &mld, ml_uint folds = 10, 
				      ml_uint cvseed = ML_DEFAULT_SEED);

  template<typename U>
  U evaluate(const ml_data &mld) const;

  ml_feature_value evaluate(const ml_instance &instance) const { return(model_.evaluate(instance)); }

  ml_string summary() const { return(model_.summary()); }

  T &model() { return model_; }
  
 private:
  T model_;
};

} // namespace puml

#include "mlmodel.tcc"

