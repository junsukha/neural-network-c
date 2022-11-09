#pragma once

#include <cstdio>
#include <vector>

#include "image.h"
#include "model.h"

namespace naivebayes {

/**
 * Tests the accuracy of a model.
 */
class Classifier {
 public:
  Classifier();
  Classifier(Model &model);

  /**
   * Checks the predicted class against the actual class for given images
   * @param images to verify
   * @return accuracy between 0 and 1
   */
  double CalculateAccuracy(const std::vector<Image> &images);

  /**
   * Calculates likelihood scores for classes 0-9 for an image
   * @param image to calculate scores for
   */
  void CalculateLikelihoodScores(const Image &image);

  /**
   * Returns the class with the highest likelihood score for an image
   * @param image to predict class for
   * @return predicted class
   */
  int GetBestClass(const Image &image);

  const double GetScore(size_t num) const;

  void SetModel(Model &model);

 private:
  Model model_;
  std::vector<double> likelihood_scores_;
};

}  // namespace naivebayes
