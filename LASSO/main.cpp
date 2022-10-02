#include "head.h"
#include "normal_random.h"
#include "LASSO_proximal_grad_inn.h"
#include "LASSO_continuation.h"
#include "LASSO_Nesterov_inn.h"

int main()
{
    int m = 500;
    int n = 1000;
    double mu=1e-3;
    MatrixXd A = randn_matrix(m,n);
    VectorXd u = randn_sparseVector(n,0.1);
    VectorXd x0= randn_vector(n);
    VectorXd b=A*u;
    cout<<"Exact solution of BP is "<<endl<<u<<endl<<endl;
    opts opts;
    opts.alpha0=step(A);
    opts.ls=1;
    opts.bb=1;
    out out;
    VectorXd xstar;

    opts.method="LASSO_proximal_grad_inn";
    xstar=LASSO_con(x0,A,b,mu,opts,&out);
    cout<<opts.method<<" solution of lasso is "<<endl<<xstar<<endl<<endl;
    cout<<out.itr_inn<<endl;
    cout<<out.itr<<endl<<endl;
/*
    opts.method="LASSO_Nesterov_inn";
    xstar=LASSO_con(x0,A,b,mu,opts,&out);
    cout<<opts.method<<" solution of lasso is "<<endl<<xstar<<endl<<endl;
    cout<<out.itr_inn<<endl;
    cout<<out.itr<<endl<<endl;
*/
    return 0;
}
