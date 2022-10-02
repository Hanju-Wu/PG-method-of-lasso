#ifndef LASSO_TYPE_H_INCLUDED
#define LASSO_TYPE_H_INCLUDED

#include <string>
struct Opts
{
  int maxit=10000;//最大迭代次数
  double ftol=1e-12;//函数值收敛条件
  double gtol=1e-6;//梯度收敛条件
  double alpha0=1e-3;//步长
  int ls=1;//是否使用线搜索
  int bb=1;//是否使用BB步长（需要同时使用线搜索）
};

struct Out
{
  double fval=0;//内层迭代结束的函数值
  int flag=0;//给外层函数判断内层是否达到收敛条件
  int itr=0;//内层迭代次数
  VectorXd fvec;
};

struct out
{
    int itr_inn;
    double fval;
    VectorXd fvec;
    int itr;
};
struct opts
{
    int maxit=30;
    int maxit_inn=200;
    double ftol=1e-8;
    double gtol=1e-6;
    double factor=0.1;
    double mu1=100;
    double alpha0=1e-3;
    double ftol_init_ratio=1e5;
    double gtol_init_ratio=1/gtol;
    double etaf=0.1;
    double etag=0.1;
    int ls=1;
    int bb=1;
    string method;
};

#endif // LASSO_TYPE_H_INCLUDED
