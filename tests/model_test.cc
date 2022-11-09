#include <core/classifier.h>
#include <core/model.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>

namespace naivebayes {

TEST_CASE("Train model") {
  Model model(5);
  std::string file = "../../../../../../tests/testing_data.txt";
  model.ParseFile(file);
  model.TrainModel();
  REQUIRE(model.GetImages().size() == 10);

  SECTION("Correct class") {
    REQUIRE(model.GetImages()[0].GetClass() == 0);
    REQUIRE(model.GetImages()[1].GetClass() == 1);
    REQUIRE(model.GetImages()[2].GetClass() == 2);
    REQUIRE(model.GetImages()[3].GetClass() == 3);
    REQUIRE(model.GetImages()[4].GetClass() == 4);
    REQUIRE(model.GetImages()[5].GetClass() == 5);
    REQUIRE(model.GetImages()[6].GetClass() == 6);
    REQUIRE(model.GetImages()[7].GetClass() == 7);
    REQUIRE(model.GetImages()[8].GetClass() == 8);
    REQUIRE(model.GetImages()[9].GetClass() == 9);
  }

  SECTION("Shade pixels") {
    REQUIRE(model.GetImages()[0].GetShade(1, 1) == 1);
    REQUIRE(model.GetImages()[0].GetShade(1, 2) == 1);
    REQUIRE(model.GetImages()[0].GetShade(2, 1) == 1);
    REQUIRE(model.GetImages()[1].GetShade(2, 2) == 1);
    REQUIRE(model.GetImages()[1].GetShade(3, 3) == 1);
    REQUIRE(model.GetImages()[3].GetShade(4, 2) == 1);
    REQUIRE(model.GetImages()[6].GetShade(0, 1) == 1);
    REQUIRE(model.GetImages()[6].GetShade(2, 2) == 1);
    REQUIRE(model.GetImages()[7].GetShade(3, 2) == 1);
    REQUIRE(model.GetImages()[9].GetShade(4, 3) == 1);
  }

  SECTION("Unshaded pixels") {
    REQUIRE(model.GetImages()[0].GetShade(0, 0) == 0);
    REQUIRE(model.GetImages()[0].GetShade(2, 2) == 0);
    REQUIRE(model.GetImages()[1].GetShade(0, 4) == 0);
    REQUIRE(model.GetImages()[2].GetShade(1, 2) == 0);
    REQUIRE(model.GetImages()[4].GetShade(1, 2) == 0);
    REQUIRE(model.GetImages()[5].GetShade(0, 4) == 0);
    REQUIRE(model.GetImages()[5].GetShade(3, 4) == 0);
    REQUIRE(model.GetImages()[8].GetShade(3, 2) == 0);
    REQUIRE(model.GetImages()[8].GetShade(1, 2) == 0);
    REQUIRE(model.GetImages()[9].GetShade(1, 2) == 0);
    REQUIRE(model.GetImages()[9].GetShade(4, 4) == 0);
  }

  SECTION("Shaded Feature count") {
    REQUIRE(model.GetFeatureCount(4, 0, 1, 0) == 0);
    REQUIRE(model.GetFeatureCount(1, 0, 1, 0) == 0);
    REQUIRE(model.GetFeatureCount(2, 1, 1, 0) == 1);
    REQUIRE(model.GetFeatureCount(1, 1, 1, 0) == 1);
    REQUIRE(model.GetFeatureCount(2, 2, 1, 1) == 1);
    REQUIRE(model.GetFeatureCount(0, 4, 1, 1) == 0);
    REQUIRE(model.GetFeatureCount(2, 3, 1, 2) == 0);
    REQUIRE(model.GetFeatureCount(1, 2, 1, 3) == 0);
    REQUIRE(model.GetFeatureCount(1, 2, 1, 5) == 0);
    REQUIRE(model.GetFeatureCount(2, 2, 1, 6) == 1);
    REQUIRE(model.GetFeatureCount(4, 0, 1, 8) == 0);
    REQUIRE(model.GetFeatureCount(4, 2, 1, 9) == 0);
    REQUIRE(model.GetFeatureCount(3, 2, 1, 9) == 0);
  }

  SECTION("Unshaded feature count") {
    REQUIRE(model.GetFeatureCount(0, 0, 0, 0) == 1);
    REQUIRE(model.GetFeatureCount(1, 3, 0, 3) == 0);
    REQUIRE(model.GetFeatureCount(3, 4, 0, 4) == 1);
    REQUIRE(model.GetFeatureCount(0, 1, 0, 5) == 0);
    REQUIRE(model.GetFeatureCount(3, 0, 0, 5) == 1);
    REQUIRE(model.GetFeatureCount(0, 2, 0, 6) == 1);
    REQUIRE(model.GetFeatureCount(1, 4, 0, 7) == 1);
    REQUIRE(model.GetFeatureCount(1, 3, 0, 7) == 0);
    REQUIRE(model.GetFeatureCount(4, 4, 0, 8) == 1);
    REQUIRE(model.GetFeatureCount(2, 2, 0, 9) == 0);
  }

  SECTION("Prior probabilities") {
    REQUIRE(model.GetPriorProbability(0) == 0.1);
    REQUIRE(model.GetPriorProbability(1) == 0.1);
    REQUIRE(model.GetPriorProbability(2) == 0.1);
    REQUIRE(model.GetPriorProbability(3) == 0.1);
    REQUIRE(model.GetPriorProbability(4) == 0.1);
    REQUIRE(model.GetPriorProbability(5) == 0.1);
    REQUIRE(model.GetPriorProbability(6) == 0.1);
    REQUIRE(model.GetPriorProbability(7) == 0.1);
    REQUIRE(model.GetPriorProbability(8) == 0.1);
    REQUIRE(model.GetPriorProbability(9) == 0.1);
  }

  SECTION("Shaded feature probabilities") {
    REQUIRE(model.GetFeatureProbability(0, 0, 1, 0) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(3, 2, 1, 1) == Approx(0.6666666667));
    REQUIRE(model.GetFeatureProbability(4, 3, 1, 2) == Approx(0.6666666667));
    REQUIRE(model.GetFeatureProbability(1, 0, 1, 3) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(0, 2, 1, 4) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(4, 0, 1, 5) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(3, 3, 1, 6) == Approx(0.6666666667));
    REQUIRE(model.GetFeatureProbability(1, 4, 1, 7) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(2, 1, 1, 8) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(2, 0, 1, 9) == Approx(0.3333333333));
  }

  SECTION("Unshaded feature probabilities") {
    REQUIRE(model.GetFeatureProbability(0, 0, 0, 0) == Approx(0.6666666667));
    REQUIRE(model.GetFeatureProbability(3, 2, 0, 1) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(0, 4, 0, 2) == Approx(0.6666666667));
    REQUIRE(model.GetFeatureProbability(4, 3, 0, 3) == Approx(0.6666666667));
    REQUIRE(model.GetFeatureProbability(2, 1, 0, 4) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(2, 2, 0, 5) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(4, 4, 0, 6) == Approx(0.6666666667));
    REQUIRE(model.GetFeatureProbability(1, 3, 0, 7) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(0, 3, 0, 8) == Approx(0.3333333333));
    REQUIRE(model.GetFeatureProbability(3, 0, 0, 9) == Approx(0.6666666667));
  }
}

TEST_CASE("Operator Overloading") {
  SECTION("Saving 5x5 model") {
    Model model(5);
    std::string parse_path = "../../../../../../tests/testing_data.txt";
    model.ParseFile(parse_path);
    model.TrainModel();

    std::string write_path = "../../../../../../tests/testing_model.txt";
    std::ofstream new_file(write_path);
    new_file << model;
    REQUIRE(new_file.is_open());
    new_file.close();
  }

  SECTION("Saving 28x28 model") {
    Model model(28);
    std::string parse_path = "../../../../../../data/training.txt";
    model.ParseFile(parse_path);
    model.TrainModel();

    std::string write = "../../../../../../data/model.txt";
    std::ofstream new_file(write);
    new_file << model;
    REQUIRE(new_file.is_open());
    new_file.close();
  }

  SECTION("Loading 5x5 model") {
    Model model(5);
    std::string load_path = "../../../../../../tests/testing_model.txt";
    std::ifstream load(load_path);
    load >> model;
    REQUIRE(model.GetPriorProbability(0) == 0.1);
    REQUIRE(model.GetPriorProbability(9) == 0.1);
    REQUIRE(model.GetFeatureProbability(0, 0, 0, 0) == Approx(0.666667));
    REQUIRE(model.GetFeatureProbability(4, 4, 1, 9) == Approx(0.333333));
  }

  SECTION("Loading 28x28 model") {
    Model model(28);
    std::string load_path = "../../../../../../data/model.txt";
    std::ifstream load(load_path);
    load >> model;
    REQUIRE(model.GetPriorProbability(0) == 0.0958084);
    REQUIRE(model.GetPriorProbability(9) == 0.099002);
    REQUIRE(model.GetFeatureProbability(0, 0, 0, 0) == 0.997921);
    REQUIRE(model.GetFeatureProbability(27, 27, 1, 9) == 0.00201207);
  }
}

}  // namespace naivebayes
