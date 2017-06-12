
#pragma once

namespace puml {

template<typename T>
template<typename U> 
ml_crossvalidation_results<U> ml_model<T>::train(const ml_data &mld, ml_uint folds,
						 ml_uint cvseed) {

  ml_crossvalidation_results<U> cv_results;

  if(U::type() != model_.type()) {
    log_error("model/results type mismatch\n");
    return(cv_results);
  }

  if(mld.empty()) {
    return(cv_results);
  }

  ml_rng rng(cvseed);
  ml_data mld_shuffle(mld);
  shuffle_vector(mld_shuffle, rng);

  folds = (folds == 0) ? 1 : folds;
  ml_uint test_size = mld_shuffle.size() / folds;

  for(ml_uint fold = 0; fold < folds; ++fold) {

    log("%d fold cross-validation (fold %d)\n", folds, fold+1);

    ml_uint test_offset = fold * test_size;
    ml_data test_fold = ml_data(mld_shuffle.begin() + test_offset, mld_shuffle.begin() + test_offset + test_size);

    ml_data training_fold;
    if(fold > 0) {
      training_fold = ml_data(mld_shuffle.begin(), mld_shuffle.begin() + test_offset);
    }
    
    if(fold != (folds - 1)) {
      training_fold.insert(training_fold.end(), mld_shuffle.begin() + test_offset + test_size, mld_shuffle.end());
    }

    if(training_fold.empty()) {
      training_fold = mld_shuffle;
    }

    model_.train(training_fold);
    U fold_results = evaluate<U>(test_fold);
    cv_results.add_fold_result(fold_results);
  }

  return(cv_results);
}


template<typename T>
template<typename U> 
U ml_model<T>::evaluate(const ml_data &mld) const {

  U results(model_.mlid(), model_.index_of_feature_to_predict());

  if(U::type() != model_.type()) {
    log_error("model/results type mismatch\n");
    return(results);
  }

  for(const auto &inst_ptr : mld) {
    ml_feature_value result = model_.evaluate(*inst_ptr);
    results.collect_result(result, *inst_ptr);
  }

  return(results);
}


} // namespace puml

