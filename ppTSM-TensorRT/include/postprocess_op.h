#pragma once
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

class Softmax
{
public:
	static void Inplace_Run(const std::vector<float>::iterator &_begin, const std::vector<float>::iterator &_end);
	virtual std::vector<float> Run(const std::vector<float>::iterator &_begin, const std::vector<float>::iterator &_end);
};
