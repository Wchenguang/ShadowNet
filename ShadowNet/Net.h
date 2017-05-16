#pragma once
#include "functions.h"
#include "Layer.h"
#include "connector.h"

#include <Eigen/Dense>

using namespace Eigen;

template <class FunctionType>
class MultiBPNet
{
private:
	int HidenLayerNum;
	int HidenConnectorNum;
	FullConectedLayher<LinerFunction> BPInputLayer;
	OutputLayer<FunctionType> BPOutputLayer;
	FullConectedLayher<FunctionType> *BPHidenLayers;
	Connector<FullConectedLayher<LinerFunction>, FullConectedLayher<FunctionType>> IConnector;
	Connector<FullConectedLayher<FunctionType>, OutputLayer<FunctionType>> OConnector;
	Connector<FullConectedLayher<FunctionType>, FullConectedLayher<FunctionType>> *HConnectors;

	MatrixXd Inputarrs;
	MatrixXd Expextarrs;
public:
	MultiBPNet() : HidenLayerNum(0), HidenConnectorNum(0), BPHidenLayers(0), HConnectors(0) {}
	~MultiBPNet()
	{
		if (BPHidenLayers) delete [] BPHidenLayers;
		if (HConnectors) delete [] HConnectors;
	}
	void Init(int hidenlayernum, int inputlayernuetronnum, int outputlayerneutronnum, int *hidenlayersnuutronnums, double learningrate = 0.9, double thresold = 0) 
	{ 
		HidenLayerNum = hidenlayernum;
		BPHidenLayers = new FullConectedLayher<FunctionType>[hidenlayernum];
		BPInputLayer.Init(inputlayernuetronnum, thresold);
		BPOutputLayer.Init(outputlayerneutronnum, thresold);
		for (int i = 0; i < hidenlayernum; ++i)
			BPHidenLayers[i].Init(hidenlayersnuutronnums[i], thresold);

		IConnector.Init(&BPInputLayer, &BPHidenLayers[0], learningrate);
		OConnector.Init(&BPHidenLayers[hidenlayernum - 1], &BPOutputLayer, learningrate);

		HidenConnectorNum = hidenlayernum - 1;
		HConnectors = new Connector<FullConectedLayher<FunctionType>, FullConectedLayher<FunctionType>>[HidenConnectorNum];

		for (int i = 0; i < HidenConnectorNum; ++i)
			HConnectors[i].Init(&BPHidenLayers[i], &BPHidenLayers[i + 1], learningrate);

	}

	void SetNet(MatrixXd *inputarrs, MatrixXd *expextarrs)
	{
		Inputarrs = *inputarrs;
		Expextarrs = *expextarrs;
		if (Inputarrs.cols() != BPInputLayer.NeutronsNum)
			std::cerr << "输入数据无法匹配网络输入" << endl;
		if (Expextarrs.cols() != BPOutputLayer.NeutronsNum)
			std::cerr << "预期数据无法匹配网络输出" << endl;
		if (Inputarrs.rows() != Expextarrs.rows())
			std::cerr << "输入数据与预期数据条目数不同" << endl;
	}


	void _Train()
	{
		int datarows = Inputarrs.rows() < Expextarrs.rows() ? Inputarrs.rows() : Expextarrs.rows();
		for (int j = 0; j < datarows; ++j)
		{
			BPInputLayer.SetInputs(&MatrixXd(Inputarrs.row(j)));
			BPOutputLayer.InitExpects(&MatrixXd(Expextarrs.row(j)));
			IConnector.Forward();
			for (int k = 0; k < HidenConnectorNum; ++k)
				HConnectors[k].Forward();
			OConnector.Forward();
			OConnector.BackForward();
			for (int k = HidenConnectorNum - 1; k >= 0; --k)
				HConnectors[k].BackForward();
			IConnector.BackForward();
		}
	}

	double GetError()
	{
		double error = 0, temp;
		for (int i = 0; i < BPOutputLayer.NeutronsNum; ++i)
		{
			temp = BPOutputLayer.Neutrons[i].GetOutput() - BPOutputLayer.Neutrons[i].Expect;
			error += temp * temp;
		}
		return 0.5 * error;
	}

	void Skip(int skiptimes)
	{
		for (int i = 0; i < skiptimes; ++i)
			_Train();
	}

	void Train(int traintimes)
	{
		for (int i = 0; i < traintimes; ++i)
			_Train();
		cout << GetError() << endl;
	}

	void TrainWithError(double error)
	{
		int n = 0;
		do
		{
			++n;
			_Train();
			cout << GetError() << endl;
		} while (error < GetError());
		cout << n << endl;
	}

	void Test()
	{
		int datarows = Inputarrs.rows() < Expextarrs.rows() ? Inputarrs.rows() : Expextarrs.rows();
		for (int j = 0; j < datarows; ++j)
		{
			BPInputLayer.SetInputs(&MatrixXd(Inputarrs.row(j)));
			BPOutputLayer.InitExpects(&MatrixXd(Expextarrs.row(j)));
			IConnector.Forward();
			for (int k = 0; k < HidenConnectorNum; ++k)
				HConnectors[k].Forward();
			OConnector.Forward();
			for (int i = 0; i < BPOutputLayer.NeutronsNum; ++i)
				cout << BPOutputLayer.Neutrons[i].GetOutput()<<':'<<BPOutputLayer.Neutrons[i].Expect << " ";
			cout << endl;
		}
	}

	void Destroy()
	{
		this->~MultiBPNet();
	}
};
