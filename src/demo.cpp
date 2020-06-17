#include <iostream>
#include "CMatrix.hpp"
#include "Cvector.hpp"
#include <vector>
#include <complex>

using namespace std;

Cvector<double> dVec[2];
Cvector<complex<double>> cVec[2];

CMatrix<double> dMat[2];
CMatrix<complex<double>> cMat[2];
int typeIdx; // 类型下标

//print部分
void printcVec(int num);
void printcMat(int num);
void printdVec(int num);
void printdMat(int num);

// 输入部分
void InputComplexVector(int num);
void InputComplexMatrix(int num);
void InputDoubleVector(int num);
void InputDoubleMatrix(int num);

// Vector 操作部分
void testVectorAdd();
void testVectorMinus();
void testVectorSMultple();
void testVectorSDivide();
void testVectorEleMultple();
void testVectorDot();
void testVectorCross();

// Matrix 基础操作部分
void testMatrixAdd();
void testMatrixMinus();
void testMatrixSMultple();
void testMatrixSDivide();
void testMatEleMultple();
void testMatTranpos();
void testMatConjug();
void testMatMat();
void testMatVec();

// Matrix 进阶操作部分
void testMatMax();
void testMatMin();
void testMatSum();
void testMatAvg();
void testMatEigenValue();
void testMatEigenVector();
void testMatInverse();
void testMatDet();
void testMatReshape();
void testMatSlice();
void testMat2CMatrix();
void testCMatrix2Mat();

// 菜单部分
void ChooseType(int num,bool isVec);
void commandVector(int idx);
void commandMatrixBasic(int idx);
void commandMatrixAdv(int idx);
void displayMenu();

void printcVec(int num)
{
    for (int i = 0; i < num; i++)
    {
        cout << "Vec" << i + 1 << ": ";
        cVec[i].print();
    }
}
void printdVec(int num)
{
    for (int i = 0; i < num; i++)
    {
        cout << "Vec" << i + 1 << ": ";
        dVec[i].print();
    }
}
void printcMat(int num)
{
    for (int i = 0; i < num; i++)
    {
        cout << "Mat" << i + 1 << ": ";
        cMat[i].print();
    }
}
void printdMat(int num)
{
    for (int i = 0; i < num; i++)
    {
        cout << "Mat" << i + 1 << ": ";
        dMat[i].print();
    }
}

// input complex Vector
void InputComplexVector(int num)
{
    int size;
    complex<double> ct;
    vector<complex<double>> ctmp;
    cout << " Vector " << num << ":" << endl;
    cout << "Please input size:";
    cin >> size;
    cout << "Please input element:";
    for (int i = 0; i < size; i++)
    {
        cin >> ct;
        ctmp.push_back(ct);
    }
    Cvector<complex<double>> t(ctmp);
    cVec[num - 1] = t;
}
void InputComplexMatrix(int num)
{
    int row, column;
    complex<double> ct;
    vector<complex<double>> ctmp;
    cout << " Matrix " << num << ":" << endl;
    cout << "Please input row:";
    cin >> row;
    cout << "Please input column:";
    cin >> column;
    cout << "Please input element:";
    for (int i = 0; i < row * column; i++)
    {
        cin >> ct;
        ctmp.push_back(ct);
    }
    CMatrix<complex<double>> t(row, column, ctmp);
    cMat[num - 1] = t;
}

//input double Vector
void InputDoubleVector(int num)
{
    int size;
    double dt;
    vector<double> dtmp;
    cout << " Vector " << num << ":" << endl;
    cout << "Please input size:";
    cin >> size;
    cout << "Please input element:";
    for (int i = 0; i < size; i++)
    {
        cin >> dt;
        dtmp.push_back(dt);
    }
    Cvector<double> t(dtmp);
    dVec[num - 1] = t;
}
void InputDoubleMatrix(int num)
{
    int row, column;
    double dt;
    vector<double> dtmp;
    cout << " Matrix " << num << ":" << endl;
    cout << "Please input row:";
    cin >> row;
    cout << "Please input column:";
    cin >> column;
    cout << "Please input element:";
    for (int i = 0; i < row * column; i++)
    {
        cin >> dt;
        dtmp.push_back(dt);
    }
    CMatrix<double> t(row, column, dtmp);
    dMat[num - 1] = t;
}
void ChooseType(int num, bool isVec)
{
    cout << "Choose Type:" << endl;
    cout << "1. double" << endl;
    cout << "2. complex double" << endl;
    cin >> typeIdx;

    do
    {
        switch (typeIdx)
        {
        case 1:
            for (int i = 1; i < num + 1; i++)
            {
                if (isVec)
                    InputDoubleVector(i);
                else
                    InputDoubleMatrix(i);
            }
            break;
        case 2:
            for (int i = 1; i < num + 1; i++)
            {
                if (isVec)
                    InputComplexVector(i);
                else
                    InputComplexMatrix(i);
            }
            break;

        default:
            cout << "You type in a wrong index!" << endl;
            break;
        }
    } while (typeIdx > 2 && typeIdx < 1);
}

void testVectorAdd()
{
    ChooseType(2, true);
    cout << "-------------PLUS-------------" << endl;
    if (typeIdx == 1)
    {
        printdVec(2);
        cout << "Result:" << endl;
        (dVec[0] + dVec[1]).print();
    }
    else
    {
        printcVec(2);
        cout << "Result:" << endl;
        (cVec[0] + cVec[1]).print();
    }
}
void testVectorMinus()
{
    ChooseType(2, true);
    cout << "------------MINUS-------------" << endl;
    if (typeIdx == 1)
    {
        printdVec(2);
        cout << "Result:" << endl;
        (dVec[0] - dVec[1]).print();
    }
    else
    {
        printcVec(2);
        cout << "Result:" << endl;
        (cVec[0] - cVec[1]).print();
    }
}

void testVectorSMultple()
{
    ChooseType(1, true);
    double s;
    cout << "Please input scaler:";
    cin >> s;
    cout << "--------Scaler Multiple--------" << endl;
    cout << "Scaler: " << s;
    if (typeIdx == 1)
    {
        printdVec(1);
        cout << "Result:" << endl;
        (s * dVec[0]).print();
    }
    else
    {
        printcVec(1);
        cout << "Result:" << endl;
        (s * cVec[0]).print();
    }
}


void testVectorSDivide()
{
    ChooseType(1, true);
    double s;
    cout << "Please input scaler:";
    cin >> s;
    cout << "--------Scaler Division--------" << endl;
    cout << "Scaler: " << s;
    if (typeIdx == 1)
    {
        printdVec(1);
        cout << "Result:" << endl;
        (dVec[0] / s).print();
    }
    else
    {
        printcVec(1);
        cout << "Result:" << endl;
        (cVec[0] / s).print();
    }
}
void testVectorEleMultple()
{
    ChooseType(2, true);
    cout << "-----Element-Wise Multiple-----" << endl;
    if (typeIdx == 1)
    {
        printdVec(2);
        cout << "Result:" << endl;
        (dVec[0].el_mult(dVec[1])).print();
    }
    else
    {
        printcVec(2);
        cout << "Result:" << endl;
        (cVec[0].el_mult(cVec[1])).print();
    }
}

void testVectorDot()
{
    ChooseType(2, true);
    cout << "---------Dot Multiple---------" << endl;
    if (typeIdx == 1)
    {
        printdVec(2);
        cout << "Result:" << endl;
        cout << (dVec[0].dot(dVec[1])) << endl;
    }
    else
    {
        printcVec(2);
        cout << "Result:" << endl;
        cout << (cVec[0].dot(cVec[1])) << endl;
    }
}
void testVectorCross()
{
    ChooseType(2, true);
    cout << "-------Cross Multiple---------" << endl;
    if (typeIdx == 1)
    {
        printdVec(2);
        cout << "Result:" << endl;
        (dVec[0].cross(dVec[1])).print();
    }
    else
    {
        printcVec(2);
        cout << "Result:" << endl;
        (cVec[0].cross(cVec[1])).print();
    }
}

void testMatrixAdd()
{
    ChooseType(2, false);
    cout << "-------------PLUS-------------" << endl;
    if (typeIdx == 1)
    {
        printdMat(2);
        cout << "Result:" << endl;
        (dMat[0] + dMat[1]).print();
    }
    else
    {
        printdMat(2);
        cout << "Result:" << endl;
        (cMat[0] + cMat[1]).print();
    }
}
void testMatrixMinus()
{
    ChooseType(2, false);
    cout << "------------Minus-------------" << endl;
    if (typeIdx == 1)
    {
        printdMat(2);
        cout << "Result:" << endl;
        (dMat[0] - dMat[1]).print();
    }
    else
    {
        printdMat(2);
        cout << "Result:" << endl;
        (cMat[0] - cMat[1]).print();
    }
}

void testMatrixSMultple()
{
    ChooseType(2, false);
    double s;
    cout << "Please input scaler:";
    cin >> s;
    cout << "--------Scaler Multiple--------" << endl;
    cout << "Scaler: " << s;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << "Result:" << endl;
        (dMat[0] * s).print();
    }
    else
    {
        printcMat(1);
        cout << "Result:" << endl;
        (cMat[0] * s).print();
    }
}
void testMatrixSDivide()
{
    ChooseType(2, false);
    double s;
    cout << "Please input scaler:";
    cin >> s;
    cout << "--------Scaler Division--------" << endl;
    cout << "Scaler: " << s;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << "Result:" << endl;
        (dMat[0] / s).print();
    }
    else
    {
        printcMat(1);
        cout << "Result:" << endl;
        (cMat[0] / s).print();
    }
}

void testMatEleMultple()
{
    ChooseType(2, false);
    cout << "-----Element-Wise Multiple-----" << endl;
    if (typeIdx == 1)
    {
        printdMat(2);
        cout << "Result:" << endl;
        (dMat[0].el_mult(dMat[1])).print();
    }
    else
    {
        printcMat(2);
        cout << "Result:" << endl;
        (cMat[0].el_mult(cMat[1])).print();
    }
}

void testMatTranpos()
{
    ChooseType(1, false);
    cout << "---------Tranposition---------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << "Result:" << endl;
        dMat[0].tranposition();
        (dMat[0]).print();
    }
    else
    {
        printcMat(1);
        cout << "Result:" << endl;
        cMat[0].tranposition();
        (cMat[0]).print();
    }
}
void testMatConjug()
{
    ChooseType(1, false);
    cout << "---------Tranposition---------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << "Result:" << endl;
        dMat[0].conjugation();
        (dMat[0]).print();
    }
    else
    {
        printcMat(1);
        cout << "Result:" << endl;
        cMat[0].conjugation();
        (cMat[0]).print();
    }
}
void testMatMat()
{
    ChooseType(2, false);
    cout << "-----Matrix-Matrix Multiple-----" << endl;
    if (typeIdx == 1)
    {
        printdMat(2);
        cout << "Result:" << endl;
        (dMat[0] * dMat[1]).print();
    }
    else
    {
        printcMat(2);
        cout << "Result:" << endl;
        (cMat[0] * cMat[1]).print();
    }
}
void testMatVec()
{
    ChooseType(1, false);
    ChooseType(1, true);
    cout << "-----Matrix-Vector Multiple-----" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        printdVec(1);
        cout << "Result:" << endl;
        (dMat[0] * dVec[1]).print();
    }
    else
    {
        printcMat(1);
        printcVec(1);
        cout << "Result:" << endl;
        (cMat[0] * cVec[1]).print();
    }
}
void testMatMax()
{
    //只支持实数
    ChooseType(1, false);
    cout << "---------------MAX--------------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << "Result:" << endl;
        cout << dMat[0].max_el() << endl;
    }
}
void testMatMin()
{
    //只支持实数
    ChooseType(1, false);
    cout << "---------------MIN--------------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << "Result:" << endl;
        cout << dMat[0].min_el() << endl;
    }
}
void testMatSum()
{
    ChooseType(1, false);
    cout << "-------------Summary------------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << "Result:" << endl;
        cout << dMat[0].sum() << endl;
    }
    else
    {
        printcMat(1);
        cout << "Result:" << endl;
        cout << cMat[0].sum() << endl;
    }
}
void testMatAvg()
{
    ChooseType(1, false);
    cout << "-------------Average------------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << "Result:" << endl;
        cout << dMat[0].avg() << endl;
    }
    else
    {
        printcMat(1);
        cout << "Result:" << endl;
        cout << cMat[0].avg() << endl;
    }
}
void testMatEigenValue()
{
    ChooseType(1, false);
    cout << "----------Eigen Value-----------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        //TODO
        // print eigenValue
    }
    else
    {
        printcMat(1);
        //TODO
        // print eigenValue
    }
}
void testMatEigenVector()
{
    ChooseType(1, false);
    cout << "----------Eigen Vector----------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        //TODO
        // print eigenVector
    }
    else
    {
        printcMat(1);
        //TODO
        // print eigenVector
    }
}
void testMatInverse()
{
    //只支持实数
    ChooseType(1, false);
    cout << "----------Eigen Vector----------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        dMat[0].inverse().print();
    }
}
void testMatDet()
{
    ChooseType(1, false);
    cout << "-----------Determinant-----------" << endl;
    if (typeIdx == 1)
    {
        printdMat(1);
        cout << det(dMat[0]) << endl;
    }
    else
    {
        printcMat(1);
        cout << det(cMat[0]) << endl;
    }
}
void testMatReshape()
{
    ChooseType(1, false);
    cout << "------------Reshape-------------" << endl;
    int r, c;
    cout << "Please input reshape row: ";
    cin >> r;
    cout << "Please input reshape column: ";
    cin >> c;
    if (typeIdx == 1)
    {
        printdMat(1);
        dMat[0].reshape(r, c).print();
    }
    else
    {
        printcMat(1);
        cMat[0].reshape(r, c).print();
    }
}
void testMatSlice()
{
    ChooseType(1, false);
    cout << "------------Slice-------------" << endl;
    int rs, cs, re, ce;
    cout << "Please input start row: ";
    cin >> rs;
    cout << "Please input end row: ";
    cin >> re;
    cout << "Please input start column: ";
    cin >> cs;
    cout << "Please input end column: ";
    cin >> ce;
    if (typeIdx == 1)
    {
        printdMat(1);
        dMat[0].slice(rs, re, cs, ce).print();
    }
    else
    {
        printcMat(1);
        cMat[0].slice(rs, re, cs, ce).print();
    }
}

void testMat2CMatrix(){
    //ChooseType(1,false);
    cout << "------------opencv Mat to CMatrix-------------" << endl;
    //cv::Mat t1 = (cv::Mat_<double>(3,3)<<9,8,7,6,5,4,3,2,1);
    int r,c;
    cout << "Please input row number: ";
    cin>>r;
    cout << "Please input column number: ";
    cin>>c;
    cv::Mat t1 ;
    t1.create(r,c,CV_8UC1);
    //double arr[r*c];
    double temp;
    cout << "Please input matrix elements: "<<endl;
    for(int i = 0; i < r;i++){
        for(int j = 0; j < c;j++){
            cin>>temp;
            t1.at<uchar>(i,j)=temp;
        }
    }

    cout<<"Matrix in opencv:"<<endl<<t1<<endl;
    CMatrix<double> t2 = Mat2CMatrix<double>(t1);
    std::cout<<"Matrix in CMatrix:"<<endl;
    t2.print();
}

void testCMatrix2Mat(){
    cout << "------------CMatrix to opencv Mat-------------" << endl;
    int r,c;
    cout << "Please input row number: ";
    cin>>r;
    cout << "Please input column number: ";
    cin>>c;
    double arr[r*c];
    cout << "Please input matrix elements: "<<endl;
    for(int i = 0; i < r;i++){
        for(int j = 0; j < c;j++){
            cin>>arr[i*c+j];
        }
    }
    CMatrix<double> test1(r, c, arr);
    cv::Mat t1 = test1.CMatirxr2Mat();
    std::cout<<"Matrix in CMatrix:"<<endl;
    test1.print();
    cout<<"Matrix in opencv:"<<endl<<t1<<endl;

}

int main()
{
    displayMenu();
}

void displayMenu()
{
    int tmp, index;
    cout << "-----------------Command Menu-----------------" << endl;
    cout << "    1. Vector" << endl;
    cout << "    2. Matrix Basic" << endl;
    cout << "    3. Matrix Advance" << endl;
    cout << "    4. Exit (Default)" << endl;
    cin >> tmp;
    if (tmp == 1)
    {
        cout << "--------------Vector Command Menu--------------" << endl;
        cout << "   1. PLUS" << endl;
        cout << "   2. MINUS" << endl;
        cout << "   3. Scaler Multiple" << endl;
        cout << "   4. Scaler Divide" << endl;
        cout << "   5. Element-wise Multiple" << endl;
        cout << "   6. Dot Mult" << endl;
        cout << "   7. Cross Mult" << endl;
        cout << "   0. Back to Main Menu" << endl;
        cout << "-----------------------------------------------" << endl;
        cout << "Please input operation index: ";
        cin >> index;
        commandVector(index);
    }
    else if (tmp == 2)
    {
        cout << "---------------Matrix Basic Menu--------------" << endl;
        cout << "   1. PLUS" << endl;
        cout << "   2. MINUS" << endl;
        cout << "   3. Scaler Multiple" << endl;
        cout << "   4. Scaler Divide" << endl;
        cout << "   5. Element-wise Multiple" << endl;
        cout << "   6. Tranposition" << endl;
        cout << "   7. Conjugation" << endl;
        cout << "   8. Matrix Multiple" << endl;
        cout << "   9. Matrix-Vector Multiple" << endl;
        cout << "   0. Back to Main Menu" << endl;
        cout << "-----------------------------------------------" << endl;
        cout << "Please input operation index: ";
        cin >> index;
        commandMatrixBasic(index);
    }
    else if (tmp == 3)
    {
        cout << "-------------Matrix Advanced Menu--------------" << endl;
        cout << "   1. Find MAX" << endl;
        cout << "   2. Find MIN" << endl;
        cout << "   3. Sum Up" << endl;
        cout << "   4. Average" << endl;
        cout << "   5. Eigen Value" << endl;
        cout << "   6. Eigen Vector" << endl;
        cout << "   7. Inverse " << endl;
        cout << "   8. Determinant" << endl;
        cout << "   9. Reshape" << endl;
        cout << "   10. Slicing" << endl;
        cout << "   11. CMatrix to opencv Mat" << endl;
        cout << "   12. opencv Mat to CMatrix" << endl;
        cout << "   0. Back to Main Menu" << endl;
        cout << "-----------------------------------------------" << endl;
        cout << "Please input operation index: ";
        cin >> index;
        commandMatrixAdv(index);
    }
    else
    {
        cout << "Exit!" << endl;
        return;
    }
}
// Vector Operation
void commandVector(int idx)
{
    switch (idx)
    {
    case 1:
        testVectorAdd();
        break;
    case 2:
        testVectorMinus();
        break;
    case 3:
        testVectorSMultple();
        break;
    case 4:
        testVectorSDivide();
        break;
    case 5:
        testVectorEleMultple();
        break;
    case 6:
        testVectorDot();
        break;
    case 7:
        testVectorCross();
        break;
    case 0:
        displayMenu();
        break;
    default:
        cout << "Wrong Index!" << endl;
        break;
    }
}
// Matrix Basic Operation
void commandMatrixBasic(int idx)
{
    switch (idx)
    {
    case 1:
        testMatrixAdd();
        break;
    case 2:
        testMatrixMinus();
        break;
    case 3:
        testMatrixSMultple();
        break;
    case 4:
        testMatrixSDivide();
        break;
    case 5:
        testMatEleMultple();
        break;
    case 6:
        testMatTranpos();
        break;
    case 7:
        testMatConjug();
        break;
    case 8:
        testMatMat();
        break;
    case 9:
        testMatVec();
        break;
    case 0:
        displayMenu();
        break;
    default:
        cout << "Wrong Index!" << endl;
        break;
    }
}
// Matrix Advanced Operation
void commandMatrixAdv(int idx)
{
    switch (idx)
    {
    case 1:
        testMatMax();
        break;
    case 2:
        testMatMin();
        break;
    case 3:
        testMatSum();
        break;
    case 4:
        testMatAvg();
        break;
    case 5:
        testMatEigenValue();
        break;
    case 6:
        testMatEigenVector();
        break;
    case 7:
        testMatInverse();
        break;
    case 8:
        testMatDet();
        break;
    case 9:
        testMatReshape();
        break;
    case 10:
        testMatSlice();
        break;
    case 11:
        testCMatrix2Mat();
        break;
    case 12:
        testMat2CMatrix();
        break;
    case 0:
        displayMenu();
        break;
    default:
        cout << "Wrong Index!" << endl;
        break;
    }
}
