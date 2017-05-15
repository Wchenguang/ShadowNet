#pragma once

#include "Neutron.h"
#include <Eigen/Dense>

using namespace Eigen;

template <class NeutronType>
class Layer
{};

template <class FunctionType>
class FullConectedLayher : public Layer<FullConectedNeutron<FunctionType>>
{
public:
	FullConectedNeutron<FunctionType> *Neutrons;
	int NeutronsNum;

	FullConectedLayher() : Neutrons(0), NeutronsNum(0) {}
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