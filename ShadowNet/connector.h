#pragma once

#include "functions.h"
#include "Layer.h"

#include <iostream>
using namespace std;

template <class LastLayerType, class NextLayerType>
class Connector
{};

template<>
class Connector<FullConectedLayher<SigmodFunction>, OutputLayer<SigmodFunction>>
{
private:
	FullConectedLayher<SigmodFunction> *LastLayer;
	OutputLayer<SigmodFunction> *NextLayer;
	double **PThresolds;
	double *BackforwardFactors;
public:
	Connector() : LastLayer(0), NextLayer(0), PThresolds(0), BackforwardFactors(0) {}
	~Connector() { if (PThresolds) delete [] PThresolds; if (BackforwardFactors) delete [] BackforwardFactors; }
	void Init(FullConectedLayher<SigmodFunction> *lastlayer, OutputLayer<SigmodFunction> *nextlayer, 
				double LearningRate = 0.01)
	{
		LastLayer = lastlayer;
		NextLayer = nextlayer;
		PThresolds = new double *[NextLayer->NeutronsNum];
		BackforwardFactors = new double[NextLayer->NeutronsNum];
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			PThresolds[i] = NextLayer->Neutrons[i].GetPThresold();
		lastlayer->_Init(BackforwardFactors, PThresolds, NextLayer->NeutronsNum, LearningRate);
	}
	void Forward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
		{
			double temp = 0;
			for (int j = 0; j < LastLayer->NeutronsNum; ++j)
				temp += LastLayer->Neutrons[j].GetOutput() * LastLayer->Neutrons[j].NextWeight[i];
			NextLayer->Neutrons[i].SetInput(temp);
			NextLayer->Neutrons[i].GetOutput();
		}
	}
	void BackForward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			BackforwardFactors[i] = NextLayer->Neutrons[i].GetBackForwardFactor();
		for (int i = 0; i < LastLayer->NeutronsNum; ++i)
			LastLayer->Neutrons[i].Update();
	}
};

template<>
class Connector<FullConectedLayher<PreluFunction>, OutputLayer<PreluFunction>>
{
private:
	FullConectedLayher<PreluFunction> *LastLayer;
	OutputLayer<PreluFunction> *NextLayer;
	double **PThresolds;
	double *BackforwardFactors;
public:
	Connector() : LastLayer(0), NextLayer(0), PThresolds(0), BackforwardFactors(0) {}
	~Connector() { if (PThresolds) delete[] PThresolds; if (BackforwardFactors) delete[] BackforwardFactors; }
	void Init(FullConectedLayher<PreluFunction> *lastlayer, OutputLayer<PreluFunction> *nextlayer,
		double LearningRate = 0.01)
	{
		LastLayer = lastlayer;
		NextLayer = nextlayer;
		PThresolds = new double *[NextLayer->NeutronsNum];
		BackforwardFactors = new double[NextLayer->NeutronsNum];
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			PThresolds[i] = NextLayer->Neutrons[i].GetPThresold();
		lastlayer->_Init(BackforwardFactors, PThresolds, NextLayer->NeutronsNum, LearningRate);
	}
	void Forward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
		{
			double temp = 0;
			for (int j = 0; j < LastLayer->NeutronsNum; ++j)
				temp += LastLayer->Neutrons[j].GetOutput() * LastLayer->Neutrons[j].NextWeight[i];
			NextLayer->Neutrons[i].SetInput(temp);
			NextLayer->Neutrons[i].GetOutput();
		}
	}
	void BackForward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			BackforwardFactors[i] = NextLayer->Neutrons[i].GetBackForwardFactor();
		for (int i = 0; i < LastLayer->NeutronsNum; ++i)
			LastLayer->Neutrons[i].Update();
	}
};

template<>
class Connector<FullConectedLayher<SigmodFunction>, FullConectedLayher<SigmodFunction>>
{
private:
	FullConectedLayher<SigmodFunction> *LastLayer;
	FullConectedLayher<SigmodFunction> *NextLayer;
	double **PThresolds;
	double *BackforwardFactors;
public:
	Connector() : LastLayer(0), NextLayer(0), PThresolds(0), BackforwardFactors(0) {}
	~Connector() { if (PThresolds) delete[] PThresolds; if (BackforwardFactors) delete[] BackforwardFactors; }
	void Init(FullConectedLayher<SigmodFunction> *lastlayer, FullConectedLayher<SigmodFunction> *nextlayer,
		double LearningRate = 0.01)
	{
		LastLayer = lastlayer;
		NextLayer = nextlayer;
		PThresolds = new double *[NextLayer->NeutronsNum];
		BackforwardFactors = new double[NextLayer->NeutronsNum];
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			PThresolds[i] = NextLayer->Neutrons[i].GetPThresold();
		lastlayer->_Init(BackforwardFactors, PThresolds, NextLayer->NeutronsNum, LearningRate);
	}
	void Forward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
		{
			double temp = 0;
			for (int j = 0; j < LastLayer->NeutronsNum; ++j)
			{
				temp += LastLayer->Neutrons[j].GetOutput() * LastLayer->Neutrons[j].NextWeight[i];
			}
			NextLayer->Neutrons[i].SetInput(temp);
			NextLayer->Neutrons[i].GetOutput();
		}
	}
	void BackForward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			BackforwardFactors[i] = NextLayer->Neutrons[i].GetBackForwardFactor();
		for (int i = 0; i < LastLayer->NeutronsNum; ++i)
			LastLayer->Neutrons[i].Update();
	}
};

template<>
class Connector<FullConectedLayher<PreluFunction>, FullConectedLayher<PreluFunction>>
{
private:
	FullConectedLayher<PreluFunction> *LastLayer;
	FullConectedLayher<PreluFunction> *NextLayer;
	double **PThresolds;
	double *BackforwardFactors;
public:
	Connector() : LastLayer(0), NextLayer(0), PThresolds(0), BackforwardFactors(0) {}
	~Connector() { if (PThresolds) delete[] PThresolds; if (BackforwardFactors) delete[] BackforwardFactors; }
	void Init(FullConectedLayher<PreluFunction> *lastlayer, FullConectedLayher<PreluFunction> *nextlayer,
		double LearningRate = 0.01)
	{
		LastLayer = lastlayer;
		NextLayer = nextlayer;
		PThresolds = new double *[NextLayer->NeutronsNum];
		BackforwardFactors = new double[NextLayer->NeutronsNum];
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			PThresolds[i] = NextLayer->Neutrons[i].GetPThresold();
		lastlayer->_Init(BackforwardFactors, PThresolds, NextLayer->NeutronsNum, LearningRate);
	}
	void Forward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
		{
			double temp = 0;
			for (int j = 0; j < LastLayer->NeutronsNum; ++j)
			{
				temp += LastLayer->Neutrons[j].GetOutput() * LastLayer->Neutrons[j].NextWeight[i];
			}
			NextLayer->Neutrons[i].SetInput(temp);
			NextLayer->Neutrons[i].GetOutput();
		}
	}
	void BackForward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			BackforwardFactors[i] = NextLayer->Neutrons[i].GetBackForwardFactor();
		for (int i = 0; i < LastLayer->NeutronsNum; ++i)
			LastLayer->Neutrons[i].Update();
	}
};


template<>
class Connector<FullConectedLayher<LinerFunction>, FullConectedLayher<SigmodFunction>>
{
private:
	FullConectedLayher<LinerFunction> *LastLayer;
	FullConectedLayher<SigmodFunction> *NextLayer;
	double **PThresolds;
	double *BackforwardFactors;
public:
	Connector() : LastLayer(0), NextLayer(0), PThresolds(0), BackforwardFactors(0) {}
	~Connector() { if (PThresolds) delete[] PThresolds; if (BackforwardFactors) delete[] BackforwardFactors; }
	void Init(FullConectedLayher<LinerFunction> *lastlayer, FullConectedLayher<SigmodFunction> *nextlayer,
		double LearningRate = 0.01)
	{
		LastLayer = lastlayer;
		NextLayer = nextlayer;
		PThresolds = new double *[NextLayer->NeutronsNum];
		BackforwardFactors = new double[NextLayer->NeutronsNum];
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			PThresolds[i] = NextLayer->Neutrons[i].GetPThresold();
		lastlayer->_Init(BackforwardFactors, PThresolds, NextLayer->NeutronsNum, LearningRate);
	}
	void Forward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
		{
			double temp = 0;
			for (int j = 0; j < LastLayer->NeutronsNum; ++j)
			{
				temp += LastLayer->Neutrons[j].GetOutput() * LastLayer->Neutrons[j].NextWeight[i];
			}
			NextLayer->Neutrons[i].SetInput(temp);
		}
	}
	void BackForward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			BackforwardFactors[i] = NextLayer->Neutrons[i].GetBackForwardFactor();
		for (int i = 0; i < LastLayer->NeutronsNum; ++i)
			LastLayer->Neutrons[i].Update();
	}
};

template<>
class Connector<FullConectedLayher<LinerFunction>, FullConectedLayher<PreluFunction>>
{
private:
	FullConectedLayher<LinerFunction> *LastLayer;
	FullConectedLayher<PreluFunction> *NextLayer;
	double **PThresolds;
	double *BackforwardFactors;
public:
	Connector() : LastLayer(0), NextLayer(0), PThresolds(0), BackforwardFactors(0) {}
	~Connector() { if (PThresolds) delete[] PThresolds; if (BackforwardFactors) delete[] BackforwardFactors; }
	void Init(FullConectedLayher<LinerFunction> *lastlayer, FullConectedLayher<PreluFunction> *nextlayer,
		double LearningRate = 0.01)
	{
		LastLayer = lastlayer;
		NextLayer = nextlayer;
		PThresolds = new double *[NextLayer->NeutronsNum];
		BackforwardFactors = new double[NextLayer->NeutronsNum];
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			PThresolds[i] = NextLayer->Neutrons[i].GetPThresold();
		lastlayer->_Init(BackforwardFactors, PThresolds, NextLayer->NeutronsNum, LearningRate);
	}
	void Forward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
		{
			double temp = 0;
			for (int j = 0; j < LastLayer->NeutronsNum; ++j)
			{
				temp += LastLayer->Neutrons[j].GetOutput() * LastLayer->Neutrons[j].NextWeight[i];
			}
			NextLayer->Neutrons[i].SetInput(temp);
		}
	}
	void BackForward()
	{
		for (int i = 0; i < NextLayer->NeutronsNum; ++i)
			BackforwardFactors[i] = NextLayer->Neutrons[i].GetBackForwardFactor();
		for (int i = 0; i < LastLayer->NeutronsNum; ++i)
			LastLayer->Neutrons[i].Update();
	}
};
