#pragma once

class obj
{};

//顶层神经元类 也是 
template <class InputType, class OutputType>
class Neutron : public obj
{
public:
	InputType	Input;
	OutputType	Output;
	InputType	Thresold;

	Neutron() : Input(0), Output(0), Thresold(0) {}
	virtual ~Neutron() {}
};

//无权不可更新 的神经元
template <class InputType, class OutputType>
class UnupdatableNWNeutron : public Neutron<InputType, OutputType>
{
};


//有权不可更新的神经元
template <class InputType, class OutputType, class WeightType>
class UnupdatableWNeutron : public Neutron<InputType, OutputType>
{
public:
	UnupdatableWNeutron() {}
	virtual ~UnupdatableWNeutron() {}
};

//有权可更新的神经元
//纯虚类 需要实现权值更新函数
template <class InputType, class OutputType, class WeightType>
class UpdatableWNeutron : public Neutron<InputType, OutputType>
{
public:
	UpdatableWNeutron() {}
	virtual ~UpdatableWNeutron() {}
};
