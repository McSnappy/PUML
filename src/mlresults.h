
#pragma once

#include "mldata.h"
#include "mlutil.h"

#include <algorithm>

namespace puml {

enum class ml_regression_metric  { mae, rmse, rmsle, custom };
enum class ml_classification_metric { accuracy, custom };

class ml_results {
 
public:

  ml_results(const ml_instance_definition &mlid, 
	     ml_uint index_of_feature_to_predict);
  virtual ~ml_results() {}

  virtual void collect_result(const ml_feature_value &prediction,
			      const ml_instance &instance) = 0;
  
  void collect_results(const ml_vector<ml_feature_value> &predictions,
		       const ml_data &mld);

  void set_custom_metric(ml_double value) { custom_metric_use_ = true; custom_metric_ = value; }
  void set_custom_metric_desc(const ml_string &desc) { custom_metric_desc_ = desc; }

 protected:

  ml_instance_definition mlid_;
  ml_uint instances_ = 0;
  ml_uint index_of_feature_to_predict_ = 0;

  bool custom_metric_use_ = false;
  ml_double custom_metric_ = 0.0;
  ml_string custom_metric_desc_ = "CUSTOM";
};


class ml_regression_results final : public ml_results {

 public:

  ml_regression_results(const ml_instance_definition &mlid,
			ml_uint index_of_feature_to_predict);

  void collect_result(const ml_feature_value &prediction, 
		      const ml_instance &instance) override;

  static ml_model_type type() { return(ml_model_type::regression); }

  ml_double value_for_metric(ml_regression_metric metric) const;
  ml_string summary() const;


 private:

  ml_double mae_metric() const;
  ml_double rmse_metric() const;
  ml_double rmsle_metric() const;

  ml_double sum_absolute_error_ = 0.0;
  ml_double sum_mean_squared_error_ = 0.0;
  ml_double sum_mean_squared_log_error_ = 0.0;

};


class ml_classification_results final : public ml_results {

 public:

  ml_classification_results(const ml_instance_definition &mlid,
			    ml_uint index_of_feature_to_predict);

  void collect_result(const ml_feature_value &prediction, 
		      const ml_instance &instance) override;

  static ml_model_type type() { return(ml_model_type::classification); }

  ml_double value_for_metric(ml_classification_metric metric) const;
  ml_string summary() const;


 private:

  ml_double accuracy_metric() const;

  ml_uint instances_correctly_classified_ = 0;
  ml_map<ml_string, ml_uint> confusion_matrix_map_;

};


template<typename A>
class ml_crossvalidation_results final {

 public:

  using const_iterator = typename ml_vector<A>::const_iterator;
  const_iterator begin() const { return(fold_results_.begin()); }
  const_iterator end() const { return(fold_results_.end()); }

  ml_uint folds() const { return(fold_results_.size()); }
  const A &fold_result(ml_uint index) const { return(fold_results_[index]); }
  void add_fold_result(const A &result) { fold_results_.push_back(result); }

  ml_string summary() {
    ml_string desc;
    ml_uint fold = 0;
    ml_string arrow = "\n-----------\n          |\n          v\n";
    for(const A &results : fold_results_) {
      desc += "\n-----------\nCV Fold: " + std::to_string(++fold) + arrow + results.summary();
    }
    return(desc);
  }

  ml_double avg_for_metric(ml_classification_metric metric) {
    return(avg_metric_([&](const A &results) { return(results.value_for_metric(metric)); }));
  }

  ml_double avg_for_metric(ml_regression_metric metric) {
    return(avg_metric_([&](const A &results) { return(results.value_for_metric(metric)); }));
  }


 private:

  ml_double avg_metric_(std::function<ml_double (const A &results)> value_func) {
    ml_double avg = 0.0;
    std::for_each(fold_results_.begin(), fold_results_.end(), [&](const A &results) { avg += value_func(results); });
    return(avg / fold_results_.size());
  }

  ml_vector<A> fold_results_;

};

} // namespace puml
