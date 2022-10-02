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
double normal_random()//������̬�ֲ������
{
    static default_random_engine e(time(0));
    static normal_distribution<double> n(0,1);
    double random_number=n(e);//��������Ϊ��������������ֲ�����
    return random_number;
}

MatrixXd randn_matrix(int i,int j)//����m*nά�������Ԫ�ط�����̬�ֲ�
{
    static default_random_engine e(time(0));
    static normal_distribution<double> n(0,1);
    MatrixXd A=MatrixXd::Zero(i,j).unaryExpr([](double dummy){return n(e);});
    return A;
}

VectorXd randn_vector(int n)//����nά���������Ԫ�ط�����̬�ֲ�
{
    VectorXd a=VectorXd::Zero(n);
    for(int i=0;i<n;i++)
    {
        a(i)=normal_random();
    }
    return a;
}

VectorXd randn_sparseVector(int n,double sparsity)//����nά��ϡ���Ϊsparsity���������������Ԫ�ط�����̬�ֲ�
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
