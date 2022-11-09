#include <core/model.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using std::vector;

namespace naivebayes {

Model::Model() = default;

Model::Model(size_t image_size) {
  kImageSize_ = image_size;
  prior_count_.resize(kNumClasses, 0);
  prior_prob_.resize(kNumClasses, 0);

  feature_count_ = vector<vector<vector<vector<size_t>>>>(
      kImageSize_,
      vector<vector<vector<size_t>>>(
          kImageSize_,
          vector<vector<size_t>>(kNumShades, vector<size_t>(kNumClasses))));

  feature_prob_ = vector<vector<vector<vector<double>>>>(
      kImageSize_,
      vector<vector<vector<double>>>(
          kImageSize_,
          vector<vector<double>>(kNumShades, vector<double>(kNumClasses))));
}

void Model::ParseFile(std::string &file_path) {
  std::ifstream input(file_path);

  if (!input) {
    return;
  }

  while (!input.eof()) {
    Image image(kImageSize_);
    input >> image;
    prior_count_[image.GetClass()] += 1;
    images_.push_back(image);
    image_count_ += 1;
  }

  input.close();
}

void Model::CalculatePriorProbabilities() {
  for (size_t num = 0; num < prior_prob_.size(); ++num) {
    prior_prob_[num] =
        (kSmoothingFactor_ + static_cast<double>(prior_count_[num])) /
        (kNumClasses * kSmoothingFactor_ + static_cast<double>(image_count_));
  }
}

void Model::TrainModel() {
  CalculatePriorProbabilities();
  CountFeatures();
  CalculateFeatureProbabilities();
}

void Model::CountFeatures() {
  for (Image &image : images_) {
    for (size_t row = 0; row < kImageSize_; ++row) {
      for (size_t col = 0; col < kImageSize_; ++col) {
        size_t shade = image.GetShade(row, col);
        feature_count_[row][col][shade][image.GetClass()] += 1;
      }
    }
  }
}

void Model::CalculateFeatureProbabilities() {
  for (size_t row = 0; row < kImageSize_; ++row) {
    for (size_t col = 0; col < kImageSize_; ++col) {
      for (size_t shade = 0; shade < kNumShades; ++shade) {
        for (size_t num = 0; num < kNumClasses; ++num) {
          feature_prob_[row][col][shade][num] =
              (kSmoothingFactor_ +
               static_cast<double>(feature_count_[row][col][shade][num])) /
              (kNumShades * kSmoothingFactor_ +
               static_cast<double>(prior_count_[num]));
        }
      }
    }
  }
}

std::ostream &operator<<(std::ostream &os, Model &model) {
  for (size_t num = 0; num < kNumClasses; ++num) {
    os << model.prior_prob_[num] << std::endl;
    for (size_t shade = 0; shade < kNumShades; ++shade) {
      for (size_t row = 0; row < model.kImageSize_; ++row) {
        for (size_t col = 0; col < model.kImageSize_; ++col) {
          os << model.feature_prob_[row][col][shade][num] << " ";
        }
        os << std::endl;
      }
    }
  }

  return os;
}

std::istream &operator>>(std::istream &is, Model &model) {
  std::string line;

  for (size_t num = 0; num < kNumClasses; ++num) {
    getline(is, line);
    model.prior_prob_[num] = std::stod(line);

    for (size_t shade = 0; shade < kNumShades; ++shade) {
      for (size_t row = 0; row < model.kImageSize_; ++row) {
        std::string feature;
        getline(is, line);
        std::istringstream ss(line);
        for (size_t col = 0; col < model.kImageSize_; ++col) {
          ss >> feature;
          model.feature_prob_[row][col][shade][num] = std::stod(feature);
        }
      }
    }
  }

  return is;
}

const std::vector<Image> &Model::GetImages() const {
  return images_;
}

size_t Model::GetFeatureCount(size_t row, size_t col, size_t shade,
                              size_t num) const {
  return feature_count_[row][col][shade][num];
}

double Model::GetFeatureProbability(size_t row, size_t col, size_t shade,
                                    size_t num) const {
  return feature_prob_[row][col][shade][num];
}

double Model::GetPriorProbability(size_t num) const {
  return prior_prob_[num];
}

}  // namespace naivebayes
