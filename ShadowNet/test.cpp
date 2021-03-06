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
	//建立训练集
	MatrixXd input(4, 2);
	input << 0, 0,
		0, 1,
		1, 0,
		1, 1;
	//建立结果集
	MatrixXd output(4, 1);
	output << 0,
		1,
		1,
		0;
	//Sigmodfunction 多层BP神经网络
	MultiBPNet<SigmodFunction> *SN = new MultiBPNet<SigmodFunction>;
	//设置神经网络参数(输入神经元个数，隐藏层个数，输出神经元个数，学习速率)以及 隐藏层的神经元数
	int *arr0 = new int[2]{ 3,4 };
	SN->Init(2, 2, 1, arr0, 0.77);
	//设置网络的训练集以及输出集
	SN->SetNet(&input, &output);
	//当误差小于0.0001时停止训练
	SN->TrainWithError(0.0001);
	//验证
	SN->Test();
	
	delete []arr0;
	delete SN;
	
	getchar();
	//Prelufunction 多层BP神经网络
	MultiBPNet<PreluFunction> PN;
	int *arr1 = new int[2]{ 3,4 };
	PN.Init(2, 2, 1, arr1, 0.07);
	PN.SetNet(&input, &output);
	//PN.Skip(100);
	PN.TrainWithError(0.000001);
	PN.Test();

	delete []arr1;
	delete PN;
	getchar();

	return 0;
}
