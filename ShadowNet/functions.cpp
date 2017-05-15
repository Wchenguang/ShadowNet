#include "functions.h"

double SigmodFunction::GetOutput(double input)
{
	return 1.0 / (1.0 + exp(-1.0*input));
}

double SigmodFunction::Derivative(double input)
{
	return input * (1 - input);
}

double PreluFunction::GetOutput(double input)
{
	if (input <= 0)
		return 0.1 * input;
	else
		return input;
}

double PreluFunction::Derivative(double input)
{
	if (input <= 0)
		return 0.1;
	else
		return 1;
}

double LinerFunction::GetOutput(double input)
{
	return input;
}

double LinerFunction::Derivative(double input)
{
	return 1;
}