#include <iostream>
#include "CMatrix.hpp"
#include "Cvector.hpp"
#include <complex>
using namespace std;

void testMatrixMulDiv()
{
    double arr2[9] = {4, 2, -5, 6, 4, -9, 5, 3, -7};
    CMatrix<double> test2(3, 3, arr2);
    CMatrix<double> test4(test2 * 3);
    CMatrix<double> test5(test2 / 3);
    test2.print();
    test2.eigenvalues_vectors();
}

int main(){
    testMatrixMulDiv();

}