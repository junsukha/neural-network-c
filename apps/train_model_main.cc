#include <core/classifier.h>
#include <core/model.h>

#include <fstream>
#include <iostream>

int main() {
  // Parses images
  naivebayes::Model model(28);
  std::string parse_path = "../data/testing.txt";
  model.ParseFile(parse_path);
  
  // Trains model and saves data
//  model.TrainModel();
//  std::string write = "../data/model.txt";
//  std::ofstream new_file(write);
//  new_file << model;
  
  // Loads data from model
  std::string file = "../data/model.txt";
  std::ifstream load_file(file);
  load_file >> model;
  
  // Calculates model accuracy
  naivebayes::Classifier classifier(model);
  std::cout << classifier.CalculateAccuracy(model.GetImages());
  
  return 0;
}
