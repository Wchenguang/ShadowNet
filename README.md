# C++神经网络库
<br/>

## 环境依赖
* Eigen<br/>

## 使用示例
* 见 test.cpp
<pre>
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
</pre>

## 实现简介
* 该网络库目前实现了BP神经网络，利用泛型化与对象化的思想，分别实现了输入层以及全联接层，并在每层之间加入了connector用于管理层与层之间误差的传递以及权重的更新