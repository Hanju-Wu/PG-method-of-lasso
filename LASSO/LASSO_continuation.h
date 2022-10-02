#ifndef LASSO_CONTINUATION_H_INCLUDED
#define LASSO_CONTINUATION_H_INCLUDED

#include "head.h"
#include "LASSO_proximal_grad_inn.h"
#include "LASSO_Nesterov_inn.h"
#include "LASSO_type.h"
#include <string>
#include <map>

typedef VectorXd (*FnPtr)(VectorXd x0,MatrixXd A,VectorXd b,double mu,double mu0,Opts opts1,Out *out1);
map<string,FnPtr> method=
{
    {"LASSO_proximal_grad_inn",LASSO_proximal_grad_inn},
    {"LASSO_Nesterov_inn",LASSO_Nesterov_inn},
};

VectorXd LASSO_con(VectorXd x0,MatrixXd A,VectorXd b,double mu0,opts opts,out *out)
{
    int k=0;
    VectorXd x=x0;
    double mu_t=opts.mu1;
    double f=funcvalue(A,b,x,mu_t);
    Opts opts1;
    opts1.ftol=opts.ftol*opts.ftol_init_ratio;
    opts1.gtol=opts.gtol*opts.gtol_init_ratio;
    opts1.ls=opts.ls;
    opts1.bb=opts.bb;
    out->itr_inn=0;
    Out out1;
    VectorXd fvec=VectorXd::Zero(opts.maxit*opts.maxit_inn);
    while (k<opts.maxit)
    {
        opts1.maxit=opts.maxit_inn;
        opts1.gtol=max(opts1.gtol*opts.etag,opts.gtol);
        opts1.ftol=max(opts1.ftol*opts.etaf,opts.ftol);
        opts1.alpha0=opts.alpha0;
        double fp=f;
        x=method[opts.method](x,A,b,mu_t,mu0,opts1,&out1);
        f=out1.fval;
        for (int i=out->itr_inn;i<out->itr_inn+out1.itr+1;i++)
        {
            fvec(i)=out1.fvec(i-out->itr_inn);
        }
        k++;
        out->itr_inn=out->itr_inn+out1.itr;
        double normG=(x-prox(1,mu0,x-gradfunc(A,b,x))).lpNorm<2>();
        if (out1.flag==0)
            mu_t=max(mu_t*opts.factor,mu0);
        if (mu_t==mu0 && (normG<opts.gtol || abs(f-fp)<opts.ftol))
            break;
    }
    out->fval=f;
    out->itr=k;
    out->fvec=fvec;
    return x;
}

#endif // LASSO_CONTINUATION_H_INCLUDED
