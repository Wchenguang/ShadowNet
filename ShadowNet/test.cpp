#include "functions.h"
#include "Neutron.h"
#include "Layer.h"
#include "connector.h"
#include <iostream>
#include <Eigen/Dense>
#include "Net.h"

//using namespace Eigen;
using namespace std;


int main()
{

	MatrixXd input(4, 2);
	input << 0, 0,
		0, 1,
		1, 0,
		1, 1;
	MatrixXd output(4, 1);
	output << 0,
		1,
		1,
		0;
	//Sigmodfunction 多层BP神经网络
	/*MultiBPNet<SigmodFunction> SN;
	SN.Init(2, 2, 1, new int[2]{ 3,4 },0.77);
	SN.SetNet(&input, &output);
	SN.TrainWithError(0.0001);
	SN.Test();

	SN.Destroy();*/

	//getchar();

	//Prelufunction 多层BP神经网络
	MultiBPNet<PreluFunction> PN;
	PN.Init(2, 2, 1, new int[2]{ 3,4 }, 0.07);
	PN.SetNet(&input, &output);
	//PN.Skip(100);
	PN.TrainWithError(0.000001);
	PN.Test();

	getchar();

	return 0;
}