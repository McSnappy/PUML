
#include "mlresults.h"

namespace puml {

ml_results::ml_results(const ml_instance_definition &mlid, 
		       ml_uint index_of_feature_to_predict) : 
  mlid_(mlid), 
  index_of_feature_to_predict_(index_of_feature_to_predict) {

}


void ml_results::collect_results(const ml_vector<ml_feature_value> &predictions,
				 const ml_data &mld) {
  if(predictions.size() != mld.size()) {
    log_error("collect_results: instance count mismatch...\n");
    return;
  }

  ml_uint inst_index = 0;
  for(const auto &prediction : predictions) {
    collect_result(prediction, *mld[inst_index++]);
  }
}


ml_regression_results::ml_regression_results(const ml_instance_definition &mlid,
					     ml_uint index_of_feature_to_predict) :
  ml_results(mlid, index_of_feature_to_predict) {
  
}


void ml_regression_results::collect_result(const ml_feature_value &prediction, 
					   const ml_instance &instance) {
					
  ml_double diff = prediction.continuous_value - instance[index_of_feature_to_predict_].continuous_value;
  sum_absolute_error_ += fabs(diff);
  sum_mean_squared_error_ += (diff * diff);
  
  ml_double ldiff = std::log(prediction.continuous_value + 1.0) - std::log(instance[index_of_feature_to_predict_].continuous_value + 1.0);
  sum_mean_squared_log_error_ += (ldiff * ldiff);
    
  ++instances_;
}


ml_double ml_regression_results::mae_metric() const {
  ml_double mae = (instances_ > 0) ? (sum_absolute_error_ / instances_) : 0.0;
  return(mae);
}


ml_double ml_regression_results::rmse_metric() const {
  ml_double rmse = (instances_ > 0) ? sqrt(sum_mean_squared_error_ / instances_) : 0.0;
  return(rmse);
}


ml_double ml_regression_results::rmsle_metric() const {
  ml_double rmsle = (instances_ > 0) ? sqrt(sum_mean_squared_log_error_ / instances_) : 0.0;
  return(rmsle);
}


ml_double ml_regression_results::value_for_metric(ml_regression_metric metric) const {
  switch(metric) {
  case ml_regression_metric::mae: return(mae_metric()); break;
  case ml_regression_metric::rmse: return(rmse_metric()); break;
  case ml_regression_metric::rmsle: return(rmsle_metric()); break;
  }

  return(0.0);
}


ml_string ml_regression_results::summary() const {
  ml_string desc = "\n*** Regression Results Summary ***\n";
  desc += "\nInstances: " + std::to_string(instances_) + "\n";
  desc += "MAE: " + std::to_string(value_for_metric(ml_regression_metric::mae)) + "\n";
  desc += "RMSE: " + std::to_string(value_for_metric(ml_regression_metric::rmse)) + "\n";
  desc += "RMSLE: " + std::to_string(value_for_metric(ml_regression_metric::rmsle)) + "\n";

  return(desc);
}


ml_classification_results::ml_classification_results(const ml_instance_definition &mlid,
						     ml_uint index_of_feature_to_predict) :
  ml_results(mlid, index_of_feature_to_predict) {
  
}


void ml_classification_results::collect_result(const ml_feature_value &prediction, 
					       const ml_instance &instance) {
					 
   ml_uint model = prediction.discrete_value_index;
   ml_uint actual = instance[index_of_feature_to_predict_].discrete_value_index;
   ml_string key = std::to_string(actual) + "-" + std::to_string(model);
   confusion_matrix_map_[key] += 1;
   ++instances_;
   if(model == actual) {
     ++instances_correctly_classified_;
   }
}


ml_double ml_classification_results::accuracy_metric() const {
  ml_float pct = (instances_ > 0) ? ((ml_float) instances_correctly_classified_ / instances_) * 100.0 : 0.0;
  return(pct);
}


ml_double ml_classification_results::value_for_metric(ml_classification_metric metric) const {
  switch(metric) {
  case ml_classification_metric::accuracy: return(accuracy_metric()); break;
  }

  return(0.0);
}


ml_string ml_classification_results::summary() const {

  ml_string desc = "\n*** Classification Results Summary ***\n";

  if(mlid_.empty()) {
    return(desc + "(invalid instance definition)\n");
  }

  if(mlid_[index_of_feature_to_predict_]->type != ml_feature_type::discrete) {
    return(desc + "(feature type mismatch)\n");
  }

  desc += "\nInstances: " + std::to_string(instances_) + "\n";
  ml_float pct = value_for_metric(ml_classification_metric::accuracy);
  desc += string_format("Correctly Classified: %d (%.1f%%)\n\n",
			instances_correctly_classified_, pct);

  // only show the confusion matrix for a reasonable number of categories
  if(mlid_[index_of_feature_to_predict_]->discrete_values.size() > 20) {
    return(desc);
  }

  for(std::size_t ii=1; ii < mlid_[index_of_feature_to_predict_]->discrete_values.size(); ++ii) {
    desc += string_format("%7c", ((char)(ii-1)) + 'a');
  }
  
  desc += "  <-- classified as\n";

  for(std::size_t ii=1; ii < mlid_[index_of_feature_to_predict_]->discrete_values.size(); ++ii) {
    for(std::size_t jj=1; jj < mlid_[index_of_feature_to_predict_]->discrete_values.size(); ++jj) {
      
      ml_string key = std::to_string(ii) + "-" + std::to_string(jj);
      ml_uint count = 0;
      ml_map<ml_string, ml_uint>::const_iterator it = confusion_matrix_map_.find(key);
      if(it != confusion_matrix_map_.end()) {
	count = it->second;
      }

      desc += string_format("%7d", count);
    }
    
    desc += string_format(" | %c = %s\n", ((char)(ii-1)) + 'a', mlid_[index_of_feature_to_predict_]->discrete_values[ii].c_str());
  }

  desc += "\n";

  return(desc);
}


} // namespace puml
