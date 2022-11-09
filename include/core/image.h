#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

#include "common.h"

namespace naivebayes {

/**
 * Stores image pixels.
 */
class Image {
 public:
  Image();
  Image(size_t image_size);

  /**
   * Stores shaded pixels in a 2d vector, with 0 being unshaded and 1 being
   * shaded
   * @param is input stream
   * @param image class
   * @return input stream
   */
  friend std::istream& operator>>(std::istream& is, Image& image);

  /**
   * Resizes vector of shades
   * @param vector_size new size of vector
   */
  void ResizeShadeVector(size_t vector_size);

  size_t GetClass() const;
  size_t GetShade(size_t row, size_t col) const;
  size_t GetImageSize() const;
  void SetShade(size_t row, size_t col, size_t shade);
  void SetImageSize(size_t image_size);
  void SetShadeVector(std::vector<std::vector<size_t>>& shades);

 private:
  size_t kImageSize_;
  size_t class_;
  std::vector<std::vector<size_t>> shades_;
};

}  // namespace naivebayes
