#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H



#include <cmath>

	//return 1.0 / (1.0 + exp(-1.0*input));
class Function
{};

class SigmodFunction : public Function
{
public:
	static double GetOutput(double input);
	static double Derivative(double input);
};

class PreluFunction : public Function
{
public:
	static double GetOutput(double input);
	static double Derivative(double input);
};

class LinerFunction
{
public:
	static double GetOutput(double input);
	static double Derivative(double input);
};

#endif // !_FUNCTIONS_H



