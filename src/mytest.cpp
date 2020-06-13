#include <iostream>
#include "CMatrix.hpp"
#include "Cvector.hpp"
#include <complex>
using namespace std;

void testMatrixMulDiv()
{
    double arr1[9] = {5,6,7,8};
    double arr2[9] = {4, 2, -5, 6, 4, -9, 5, 3, -7};
    //double arr3[9] = {-1, 1, 0, -4, 3, 0, 1, 0, 2};
    double arr3[9] = {1, 2, 3, 2, 1, 3, 3, 3, 6};
    CMatrix<double> test1(2, 2, arr1);
    CMatrix<double> test2(3, 3, arr2);
    CMatrix<double> test3(3, 3, arr3);
    test2.print();
    test2.eigenvalues_vectors();
    test3.print();
    test3.eigenvalues_vectors();
}

int main(){
    testMatrixMulDiv();

}