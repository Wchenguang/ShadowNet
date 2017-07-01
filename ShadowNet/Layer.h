#pragma once

#include "Neutron.h"
#include <Eigen/Dense>

using namespace Eigen;

//所有层的父类
template <class NeutronType>
class Layer
{};

//全连接层 也可以作为输入
template <class FunctionType>
class FullConectedLayher : public Layer<FullConectedNeutron<FunctionType>>
{
public:
	FullConectedNeutron<FunctionType> *Neutrons;	//神经元指针
	int NeutronsNum;								//神经元个数

	FullConectedLayher() : Neutrons(0), NeutronsNum(0) {}
	//个数与阈值
	void Init(int nuetronsnum, double thresold = 0)
	{
		NeutronsNum = nuetronsnum;
		Neutrons = new FullConectedNeutron<FunctionType>[nuetronsnum];
		if (!thresold)
			for (int i = 0; i < nuetronsnum; ++i)
				Neutrons[i].InitThresold(thresold);
	}
	void _Init(double *recievedfactor,
		double **nextthresold, int nextlayerneutronnum, double learningrate)
	{
		for (int i = 0; i < NeutronsNum; ++i)
			Neutrons[i]._Init(learningrate, recievedfactor, nextthresold, nextlayerneutronnum);
	}
	void SetInputs(double *inputs)
	{
		for (int i = 0; i < NeutronsNum; ++i)
			Neutrons[i].SetInput(inputs[i]);
	}
	void SetInputs(MatrixXd *inputs)
	{
		for (int i = 0; i < NeutronsNum; ++i)
			Neutrons[i].SetInput((*inputs)(0, i));
	}
	~FullConectedLayher() { if (Neutrons) delete [] Neutrons; }
};

//输出层
template <class FunctionType>
//typedef int FunctionType;
class OutputLayer : public Layer<OutputNeutron<FunctionType>>
{
public:
	OutputNeutron<FunctionType> *Neutrons;
	int NeutronsNum;

	OutputLayer() : Neutrons(0), NeutronsNum(0) {}
	void Init(int nuetronsnum, double thresold = 0)
	{
		NeutronsNum = nuetronsnum;
		Neutrons = new OutputNeutron<FunctionType>[nuetronsnum];
		if (!thresold)
			for (int i = 0; i < nuetronsnum; ++i)
				Neutrons[i].InitThresold(thresold);
	}
	void InitExpects(double *expects)
	{
		for (int i = 0; i < NeutronsNum; ++i)
			Neutrons[i].InitExpect(expects[i]);
	}
	void InitExpects(MatrixXd *expects)
	{
		for (int i = 0; i < NeutronsNum; ++i)
			Neutrons[i].InitExpect((*expects)(0, i));
	}

	~OutputLayer() { if (Neutrons) delete [] Neutrons; }
};
