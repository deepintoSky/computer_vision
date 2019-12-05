#include <iostream>
#include <math.h>
#include <algorithm>  
#include <stdlib.h>
// Eigen 部分
#include "Eigen/Eigen"
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense> 

using namespace std;
using namespace Eigen;
 
int main( int argc, char** argv  )
{
	MatrixXd p(2, 1), p_optim(2, 1), p_update(2, 1), h(2, 1), w(2, 2); 
	MatrixXd x(63, 1), y(63, 1), y_hat(63, 1);
	MatrixXd jeco(63, 2), err(63, 1), err_update(63, 1);

	float lambda = 1, err_optim;
	int itrator = 0, ep = 100;
	int a, b;
/***********变量输入及初始化************************/
	x << 0.000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000,
	    	1.000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000,
	    	2.000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000, 2.7000, 2.8000, 2.9000,
	    	3.000, 3.1000, 3.2000, 3.3000, 3.4000, 3.5000, 3.6000, 3.7000, 3.8000, 3.9000,
	    	4.000, 4.1000, 4.2000, 4.3000, 4.4000, 4.5000, 4.6000, 4.7000, 4.8000, 4.9000,
	    	5.000, 5.1000, 5.2000, 5.3000, 5.4000, 5.5000, 5.6000, 5.7000, 5.8000, 5.9000,
	    	6.000, 6.1000, 6.2000;



	y << 84.8813,   146.7663,  149.6254, 93.1977,   3.6583,   -104.9392, -177.0434, -150.9106, -60.8202, 48.5765,
	    	154.1925,  206.0641,  151.7364, 63.3721,  -55.5942, -158.1013, -164.6986, -109.7675, 0.1444,   90.0419, 
	    	141.2188,  146.5047,  81.6069,  -14.3515, -70.5310, -96.3240,  -59.9917,  -5.2409,   52.9212,  62.2246,
	    	56.0940,   0.4309,    -25.9245, -35.8837, -6.8627,  44.8732,   91.0407,   82.4212,   56.4174,  -13.5477,
	    	-101.5125, -114.6539, -84.9267, 2.7138,   87.8119,  173.4679,  175.2198,  99.4924,   -9.1194,  -109.7865,
	    	-168.8934, -164.3817, -66.9185, 50.9492,  167.1449, 192.5162,  161.2291,  48.3567,   -53.2931, -44.5062,
	    	142.7662, -99.8187,  6.7414;

	//p(0) = 110;
	//p(1) = 110;
	
	w << 1, 0, 0, 1;
	p_optim = p;
	err_optim = 1000000.0;
/**************************************************/

/***********中间变量初始化************************/
	

/**************************************************/

	for (a = 50; a < 150; a += 1){
	for (b = 50; b < 150; b += 1){

	itrator = 0;
	p << a, b;
	h << 1, 1;
	while((h.norm() > 0.01) && (itrator < 50)){
		itrator++; 

		for (int i = 0; i < 63; i++){
			y_hat(i) = p(0) * cos(p(1) * x(i)) + p(1) * sin(p(0) * x(i));//计算拟合
			err(i) = y(i) - y_hat(i);//计算误差
			jeco(i, 0) =  cos(p(1) * x(i)) + (p(1) * x(i)) * cos(p(0) * x(i));
			jeco(i, 1) =  sin(p(0) * x(i)) - (p(0) * x(i)) * sin(p(1) * x(i));//计算雅克比


		}
		//factor = lambda * diag( jeco.transpose() * jeco ); 
		h = - (jeco.transpose() * jeco  + lambda * w).inverse() * jeco.transpose() * err;//计算h
		p_update = p + h;
		for (int i = 0; i < 63; i++){
			y_hat(i) = p_update(0) * cos(p_update(1) * x(i)) + p_update(1) * sin(p_update(0) * x(i));//计算拟合
			err_update(i) = y(i) - y_hat(i);//计算误差
			
		}

		double rho = (err_update - err).norm() / (jeco * h).norm();
		//cout << "rho is " << rho << endl;

		if (rho > 0.75){
			lambda *= 0.5;	
					
		}
   		else if(rho < 0.25){
			lambda *= 2;
   		}
		else{
			p = p + h;
			//cout << "step is " << h(0) << '	' << h(1) << endl;
			//cout << "err is " << err.norm() << endl;
			//cout << "optimation of p is " << p(0) << '	' << p(1) << endl;
			//cout << "lambda is " << lambda << endl;
		}
			
	}

	if( err.norm() <= err_optim ){
		p_optim = p;
		err_optim = err.norm();
	}
	cout << "length of itrator is " << itrator << endl;
	cout << "err is " << err_optim << endl;
	cout << "optimation of p is " << p_optim(0) << '	' << p_optim(1) << endl;

	}}
			 
	return 0;
}

