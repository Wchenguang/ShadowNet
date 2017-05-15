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
	MultiBPNet n;
	n.Init(2, 2, 1, new int[2]{ 3,4 },0.77);
	n.SetNet(&input, &output);
	//n.Train(1400);
	n.TrainWithError(0.0001);

	n.Test();
	//cout << output(0, 0);

	
		

	getchar();

	return 0;
}