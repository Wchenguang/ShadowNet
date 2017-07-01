/*
*用于链接不同层的Connector
*Net通过控制Connector控制层间的正向传播 与 反向传播
*每个Layer必须实现 Foward 和 BackForward
*代码复用性差--无法解决Layer泛型化的同时其模板参数 function的泛型化
*/


#pragma once

#include "functions.h"
#include "Layer.h"

#include <iostream>
using namespace std;


//所有Connector的父类
template <class LastLayerType, class NextLayerType>
class Connector
{};

//链接 SigmodFunction类型的 全连接层和输出层
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

//链接 PReluFunction类型的 全连接层和输出层
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


//链接 SigmodFunction类型的 全连接层和全连接层
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

//链接 PReluFunction类型的 全连接层和输出层
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

//链接 LinerFunction类型的 全连接层和全连接层
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

//链接 SigmodFunction类型的全连接层和PReluFunctino的全连接层
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
