#ifndef HEAD_H_INCLUDED
#define HEAD_H_INCLUDED

#include <iostream>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include "maxeigvalue.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;
using namespace Eigen;



int sign(double x)//符号函数
{
    if(x>0)
        return 1;
    else if(x==0)
        return 0;
    else
        return -1;
}
VectorXd prox(double t,double mu,VectorXd x)//邻近算子
{
    int n=x.size();
    VectorXd xp=VectorXd::Zero(n);
    for(int i=0;i<n;i++)
    {
        xp(i)=sign(x(i))*max(abs(x(i))-t*mu,double(0));
    }
    return xp;
}
VectorXd gradfunc(MatrixXd A,VectorXd b,VectorXd x)//梯度向量
{
    return (A.transpose())*(A*x-b);
}
double funcvalue(MatrixXd A,VectorXd b,VectorXd x,double mu)//计算x点处的函数值
{
    return mu*(x.lpNorm<1>())+0.5*(A*x-b).squaredNorm();
}

double step(MatrixXd A)//计算固定步长
{
    double L;
    if(A.rows()<A.cols())
        L=maxeigvalue(A*A.transpose());
    else
        L=maxeigvalue(A.transpose()*A);
    double t=1/L;
    return t;
}


#endif // HEAD_H_INCLUDED
