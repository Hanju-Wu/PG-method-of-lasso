#ifndef LASSO_TYPE_H_INCLUDED
#define LASSO_TYPE_H_INCLUDED

#include <string>
struct Opts
{
  int maxit=10000;//����������
  double ftol=1e-12;//����ֵ��������
  double gtol=1e-6;//�ݶ���������
  double alpha0=1e-3;//����
  int ls=1;//�Ƿ�ʹ��������
  int bb=1;//�Ƿ�ʹ��BB��������Ҫͬʱʹ����������
};

struct Out
{
  double fval=0;//�ڲ���������ĺ���ֵ
  int flag=0;//����㺯���ж��ڲ��Ƿ�ﵽ��������
  int itr=0;//�ڲ��������
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
