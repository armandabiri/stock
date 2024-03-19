// C++ program for the above approach
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

// Driver code
int main(int argc, char** argv) {
  std::cout << "Start opencv test5...." << std::endl;
  // Read the image file as
  std::string filename = "C:/repos/stock/cpp/sources/mouse.jpg";
  cv::Mat image = cv::imread(filename);

  // std::cout << "File readed!" << std::endl;

  // if (image.empty()) {
  //   std::cout << "Image File " << "Not Found" << std::endl;
  // } else {
  //   std::cout << "Image File " << "Found" << std::endl;
  // }

  // // Show Image inside a window with
  // // the name provided
  // cv::imshow("Window Name", image);

  // // Wait for any keystroke
  // cv::waitKey(0);
  return 0;
}