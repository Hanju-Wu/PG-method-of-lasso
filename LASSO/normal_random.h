#ifndef NORMAL_RANDOM_H_INCLUDED
#define NORMAL_RANDOM_H_INCLUDED

#include <functional>
#include <ctime>
#include <random>
#include <Eigen/Eigen>
#include <algorithm>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
double normal_random()//生成正态分布随机数
{
    static default_random_engine e(time(0));
    static normal_distribution<double> n(0,1);
    double random_number=n(e);//把引擎作为参数，调用随机分布对象
    return random_number;
}

MatrixXd randn_matrix(int i,int j)//生成m*n维随机矩阵，元素服从正态分布
{
    static default_random_engine e(time(0));
    static normal_distribution<double> n(0,1);
    MatrixXd A=MatrixXd::Zero(i,j).unaryExpr([](double dummy){return n(e);});
    return A;
}

VectorXd randn_vector(int n)//生成n维随机向量，元素服从正态分布
{
    VectorXd a=VectorXd::Zero(n);
    for(int i=0;i<n;i++)
    {
        a(i)=normal_random();
    }
    return a;
}

VectorXd randn_sparseVector(int n,double sparsity)//生成n维，稀疏度为sparsity的随机向量，非零元素服从正态分布
{
    int ans=n*sparsity;
    VectorXi index=VectorXi::Zero(n);;
    VectorXd normal_spVec=VectorXd::Zero(n);
    srand((unsigned)time(NULL));
    for(int i=0;i<n;i++)
    {
        index(i)=i;
    }
    random_shuffle(index.begin(),index.end());
    for(int i=0;i<ans;i++)
    {
        normal_spVec(index(i))=normal_random();
    }
    return normal_spVec;
}

#endif // NORMAL_RANDOM_H_INCLUDED
