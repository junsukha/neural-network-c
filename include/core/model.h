#pragma once

#include <array>
#include <string>
#include <vector>

#include "common.h"
#include "image.h"

namespace naivebayes {

/**
 * A model which does all the Naive Bayes calculations.
 */
class Model {
 public:
  Model();
  Model(size_t image_size);

  /**
   * Loads in feature probabilities and prior probabilities
   * @param os output stream
   * @param model to load
   * @return output stream
   */
  friend std::ostream& operator<<(std::ostream& os, Model& model);

  /**
   * Reads feature probabilities and prior probabilities
   * @param is input stream
   * @param model to read
   * @return input stream
   */
  friend std::istream& operator>>(std::istream& is, Model& model);

  /**
   * Reads given file and stores images and priors
   * @param file_path given file
   */
  void ParseFile(std::string& file_path);

  /**
   * Calculates prior probabilities
   */
  void CalculatePriorProbabilities();

  /**
   * Calculations needed for Bayes' Theorem
   */
  void TrainModel();

  /**
   * Gets a count of the number of images belonging to a class where a pixel is
   * un/shaded
   */
  void CountFeatures();

  /**
   * Calculates feature probabilities
   */
  void CalculateFeatureProbabilities();

  size_t GetFeatureCount(size_t row, size_t col, size_t shade,
                         size_t num) const;
  double GetPriorProbability(size_t num) const;
  double GetFeatureProbability(size_t row, size_t col, size_t shade,
                               size_t num) const;
  const std::vector<Image> &GetImages() const;

 private:
  size_t kImageSize_;
  size_t image_count_ = 0;
  // Can't be a const or else copy assignment operator won't work
  double kSmoothingFactor_ = 1.0;

  std::vector<Image> images_;
  std::vector<std::vector<std::vector<std::vector<size_t>>>> feature_count_;
  // (row, column, shade, number)
  std::vector<std::vector<std::vector<std::vector<double>>>> feature_prob_;
  // (row, column, shade, number)
  std::vector<size_t> prior_count_;  // (number)
  std::vector<double> prior_prob_;   // (number)
};

}  // namespace naivebayes
