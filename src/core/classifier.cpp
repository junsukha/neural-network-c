#include "core/classifier.h"

namespace naivebayes {

Classifier::Classifier() = default;

Classifier::Classifier(Model& model) : model_(model) {}

double Classifier::CalculateAccuracy(const std::vector<Image>& images) {
  size_t count = 0;
  for (const auto& image : images) {
    size_t predicted = GetBestClass(const_cast<Image&>(image));
    size_t actual = image.GetClass();
    if (actual == predicted) {
      ++count;
    }
  }

  return static_cast<double>(count) / static_cast<double>(images.size());
}

void Classifier::CalculateLikelihoodScores(const Image& image) {
  likelihood_scores_.resize(kNumClasses, 0);

  for (size_t num = 0; num < kNumClasses; ++num) {
    double likelihood = log(model_.GetPriorProbability(num));
    for (size_t row = 0; row < image.GetImageSize(); ++row) {
      for (size_t col = 0; col < image.GetImageSize(); ++col) {
        size_t shade = image.GetShade(row, col);
        likelihood += log(model_.GetFeatureProbability(row, col, shade, num));
      }
    }
    likelihood_scores_[num] = likelihood;
  }
}

int Classifier::GetBestClass(const Image &image) {
  CalculateLikelihoodScores(const_cast<Image&>(image));
  int best_class = 0;
  for (int num = 1; num < kNumClasses; ++num) {
    if (likelihood_scores_[num] > likelihood_scores_[best_class]) {
      best_class = num;
    }
  }

  return best_class;
}

const double Classifier::GetScore(size_t num) const {
  return likelihood_scores_[num];
}

void Classifier::SetModel(Model& model) {
  model_ = model;
}
}  // namespace naivebayes
