#include "postprocess_op.h"

void Softmax::Inplace_Run(const std::vector<float>::iterator &_begin, const std::vector<float>::iterator &_end)
{
	const float max_value = *std::max_element(_begin, _end);
	float denominator = 1e-5;
	for (auto it = _begin; it != _end; ++it)
	{
		*it = std::exp((*it) - max_value);
		denominator += (*it);
	}
	for (auto it = _begin; it != _end; ++it)
	{
		*it /= denominator;
	}
}
std::vector<float> Softmax::Run(const std::vector<float>::iterator &_begin, const std::vector<float>::iterator &_end)
{
	std::vector<float> prob(_begin, _end);
	const float max_value = *std::max_element(prob.begin(), prob.end());
	float denominator = 0.0f;
	for (auto it = _begin, it_p = prob.begin(); it != _end; ++it, ++it_p)
	{
		(*it_p) = std::exp((*it) - max_value);
		denominator += (*it_p);
	}
	for (auto it = prob.begin(); it != prob.end(); ++it)
	{
		(*it) /= denominator;
	}
	return prob;
}