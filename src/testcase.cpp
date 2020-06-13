#include <iostream>
#include "CMatrix.hpp"
#include "Cvector.hpp"
#include <complex>
using namespace std;

double arr[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
CMatrix<double> test1(3, 3, arr);

void testMatrixAddSub()
{
    double arr2[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    CMatrix<double> test3(3, 3, arr2);
    CMatrix<double> test4(test1 + test3);
    CMatrix<double> test5(test1 - test3);
    test4.print();
    test5.print();
}

void testMatrixMulDiv()
{
    double arr2[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    CMatrix<double> test2(3, 3, arr2);
    CMatrix<double> test4(test2 * 3);
    CMatrix<double> test5(test2 / 3);
    test4.print();
    test5.print();
}

void testMatrixTrans()
{
    double arr2[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    CMatrix<double> test1(2, 5, arr2);
    test1.print();
    test1.tranposition();
    test1.print();
}

void testMatrixConj()
{
    complex<double> tar[4];
    tar[0] = {1, 2};
    tar[1] = {2, 3};
    tar[2] = {3, 4};
    tar[3] = {4, 5};
    cout << conj(tar[3]) << endl;
    CMatrix<complex<double>> test1(2, 2, tar);
    double arr2[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    CMatrix<double> test2(2, 5, arr2);
    // complex
    cout << "Complex:" << endl;
    test1.print();
    test1.conjugation();
    test1.print();
    // non-complex
    cout << "Non-Complex:" << endl;
    test2.print();
    test2.conjugation();
    test2.print();
}
void testMatrixEleMult()
{
    double arr2[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    CMatrix<double> test2(2, 5, arr2);
    test2.print();
    test2 = test2.el_mult(test2);
    test2.print();
}
void testMatrixMult()
{
    test1.print();
    test1 = test1 * test1;
    test1.print();
}
void testMatrixVector()
{
    double arr2[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    CMatrix<double> test1(3, 3, arr2);
    vector<double> arr;
    arr.push_back(2);
    arr.push_back(3);
    arr.push_back(4);
    Cvector<double> test2(arr);
    test2.print();
    test1.print();
    test1 = test1 * test2;
    test1.print();
}
void testMatrixMaxMinSum()
{
    cout << "Max in all elements: " << test1.max_el() << endl;
    cout << "Max in row #2: " << test1.max_el(true, 2) << endl;
    cout << "Min in all elements: " << test1.min_el() << endl;
    cout << "Min in column #1: " << test1.min_el(false, 1) << endl;
    cout << "Sum of the elements: " << test1.sum() << endl;
}
void testMatrixAvg()
{
    cout << "Average in all elements: " << test1.avg() << endl;
    cout << "Average in row #2: " << test1.avg(true, 2) << endl;
    cout << "Average in column #1: " << test1.avg(false, 1) << endl;
}
void testMatrixDet()
{
    cout << det(test1) << endl;
    double arr2[16] = {4, 9, 3, 2, 4, 3, 2, 1,4, 3, 3, 7, 4, 3, 7, 1,};
    CMatrix<double> test2(4,4,arr2);
    cout << det(test2) << endl;
}
void testMatrixEigen(){
	//testcase1
	double a[16] = { 4, -30, 60, -35,-30, 300, -675, 420,60, -675, 1620, -1050, -35, 420, -1050, 700};
	CMatrix<double> test2(4,4,a);
	CMatrix<double> eigenVector(4,4);
	double eigenValue[4] = {0};
	test2.eigen(&eigenVector,&eigenValue[0],1e-10);
	test2.print();
	eigenVector.print();
	cout<<endl;
	for(int i = 0; i <4;i++){
		cout<<eigenValue[i]<<" ";
	}
	cout<<endl;
	//testcase2
	/*double b[9] = { 1.23,2.12,-4.2,2.12,-5.6,8.79,-4.2,8.79,7.3};
	CMatrix<double> test3(3,3,b);
	CMatrix<double> eigenVector(3,3);
	double eigenValue[3] = {0};
	test3.eigen(&eigenVector,&eigenValue[0],1e-10);
	 test3.print();
	eigenVector.print();
	cout<<endl;
	for(int i = 0; i <3;i++){
		cout<<eigenValue[i]<<" ";
	}
	cout<<endl;*/
}
void testMatrixInverse(){
	double b[9] = {1,2,3,2,2,1,3,4,3};
	CMatrix<double> test3(3,3,b);
	CMatrix<double> inverse = test3.inverse();
	inverse.print();
}

void testMatrixReshape(){
	test1.print();
	CMatrix<double> test2 = test1.reshape(1,9);
	test2.print();
}

void testMatrixSlice(){
	test1.print();
	CMatrix<double> test2 = test1.slice(0,2,1,2);
	test2.print();
}

void testMatrixConvolution(){
	double pic[12] = {1,2,3,4,5,6,7,8,9,8,7,6};
	double ker[9] = {1,2,3,-1,0,1,2,1,2};
	CMatrix<double> picture(3,4,pic);
	CMatrix<double> kernal(3,3,ker);
	picture.print();
	kernal.print();
	CMatrix<double> ans = picture.convolution(kernal);
	ans.print();

}

void testeigenvalues_vectors()
{
    double arr1[9] = {1,-2,1,2};
    double arr2[9] = {4, 2, -5, 6, 4, -9, 5, 3, -7};
    //double arr3[9] = {-1, 1, 0, -4, 3, 0, 1, 0, 2};
    double arr3[9] = {1, 2, 3, 2, 1, 3, 3, 3, 6};
    CMatrix<double> test1(2, 2, arr1);
    CMatrix<double> test2(3, 3, arr2);
    CMatrix<double> test3(3, 3, arr3);
    test2.print();
    test2.eigenvalues_vectors();
    test1.print();
    test1.eigenvalues_vectors();
}

void testMatrix()
{
    // testMatrixAddSub();
    // testMatrixMulDiv();
    // testMatrixTrans();
    // testMatrixConj();
    // testMatrixEleMult();
    // testMatrixMult();
    // testMatrixVector();
    // testMatrixMaxMinSum();
    // testMatrixAvg();
    //testMatrixDet();
    // testMatrixEigen();
	//testMatrixInverse();
	//testMatrixReshape();
	//testMatrixSlice();
	//testMatrixConvolution();
    testeigenvalues_vectors();
}

int main()
{
    testMatrix();
    // cout << "Ended!"<< endl;
}
