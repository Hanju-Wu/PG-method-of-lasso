#ifndef LASSO_PROXIMAL_GRAD_INN_H_INCLUDED
#define LASSO_PROXIMAL_GRAD_INN_H_INCLUDED

#include "head.h"
#include "LASSO_type.h"

VectorXd LASSO_proximal_grad_inn(VectorXd x0,MatrixXd A,VectorXd b,double mu,double mu0,Opts opts1,Out *out1)
{
    int k=0;
    VectorXd x=x0;
    double t=opts1.alpha0;
    double fp=-1;
    double tempf=funcvalue(A,b,x,mu);
    double f=funcvalue(A,b,x,mu0);
    VectorXd g=gradfunc(A,b,x);
    VectorXd fvec1=VectorXd::Zero(opts1.maxit+1);
    fvec1(0)=f;
    double normG=(x-prox(1,mu,x-g)).lpNorm<2>();
    double Cval=tempf;
    double Q=1;
    double Qp;
    double gamma=0.85;
    double rhols=1e-6;
    VectorXd gp;
    VectorXd xp;
    VectorXd dx;
    VectorXd dg;
    while (k<opts1.maxit && normG>opts1.gtol && abs(f-fp)>opts1.ftol)
    {
        gp=g;
        fp=f;
        xp=x;
        x=prox(t,mu,xp-t*gp);
        if (opts1.ls)
        {
            int nls=0;
            while (1)
            {
                tempf=funcvalue(A,b,x,mu);
                if (tempf<=Cval-rhols*0.5/t*(x-xp).squaredNorm() || nls==5)
                    break;
                t=0.2*t;
                nls++;
                x=prox(t,mu,xp-t*gp);
            }
            f=funcvalue(A,b,x,mu0);
            Qp=Q;
            Q=gamma*Qp+1;
            Cval=(gamma*Qp*Cval+tempf)/Q;
        }
        else
        {
            f=funcvalue(A,b,x,mu0);
        }
        normG=(x-xp).lpNorm<2>()/t;
        g=gradfunc(A,b,x);
        if (opts1.bb && opts1.ls)
        {
            dx=x-xp;
            dg=g-gp;
            double dxg=abs(dx.transpose()*dg);
            if (dxg>0)
            {
                if (k%2==0)
                    t=dx.squaredNorm()/dxg;
                else
                    t=dxg/dg.squaredNorm();
            }
            t=min(max(t,opts1.alpha0),1e12);
        }
        else
            t=opts1.alpha0;
        k++;
        fvec1(k)=f;
    }
    if (k == opts1.maxit)
        out1->flag = 1;
    else
        out1->flag = 0;
    out1->itr=k;
    out1->fval=f;
    out1->fvec=fvec1;
    return x;
}

#endif // LASSO_PROXIMAL_GRAD_INN_H_INCLUDED
