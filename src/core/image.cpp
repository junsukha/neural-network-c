#include "core/image.h"

using std::vector;

namespace naivebayes {

Image::Image() = default;

Image::Image(size_t image_size) {
  kImageSize_ = image_size;
  shades_.resize(image_size, std::vector<size_t>(image_size, 0));
}

std::istream &operator>>(std::istream &is, Image &image) {
  const vector<char> shaded_chars = {'+', '#'};

  std::string digit;
  getline(is, digit);
  image.class_ = std::stoi(digit);

  std::string line;
  for (size_t row = 0; row < image.kImageSize_; ++row) {
    getline(is, line);
    for (size_t col = 0; col < image.kImageSize_; ++col) {
      if (std::count(shaded_chars.begin(), shaded_chars.end(), line[col])) {
        image.shades_[row][col] = 1;
      } else {
        image.shades_[row][col] = 0;
      }
    }
  }

  return is;
}

size_t Image::GetClass() const {
  return class_;
}

size_t Image::GetShade(size_t row, size_t col) const {
  return shades_[row][col];
}

size_t Image::GetImageSize() const {
  return kImageSize_;
}

void Image::SetImageSize(size_t image_size) {
  kImageSize_ = image_size;
}

void Image::SetShade(size_t row, size_t col, size_t shade) {
  shades_[row][col] = shade;
}

void Image::ResizeShadeVector(size_t vector_size) {
  shades_.resize(vector_size, vector<size_t>(vector_size, 0));
}

void Image::SetShadeVector(vector<vector<size_t>> &shades) {
  shades_ = shades;
}

}  // namespace naivebayes
