#ifndef MAXEIGVALUE_H_INCLUDED
#define MAXEIGVALUE_H_INCLUDED

#include <Eigen/Eigen>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;

double maxeigvalue(MatrixXd A)
{
    int n;
    VectorXd u0;
    double mu0;
    VectorXd y;
    double mu1;
    VectorXd u1;
    if (A.rows() == A.cols())
    {
        n = A.rows();
        u0 = VectorXd::Ones(n);
        mu0 = 0;
        y = A * u0;
        mu1 = y.lpNorm<Infinity>();
        u1 = y / mu1;
        while ((u0 - u1).lpNorm<Infinity>() > 1e-10 && abs(mu0 - mu1) > 1e-10)
        {
            mu0 = mu1;
            y = A * u1;
            u0 = u1;
            mu1 = y.lpNorm<Infinity>();
            u1 = y / mu1;
        }
    }
    return mu1;
}


#endif // MAXEIGVALUE_H_INCLUDED
