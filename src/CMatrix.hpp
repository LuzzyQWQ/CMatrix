#ifndef CMATRIX_HPP
#define CMATRIX_HPP
#include <iostream>
#include <vector>
#include <complex>
#include <typeinfo>
#include "Cvector.hpp"
#include "Cexception.hpp"
// conjugation
template <typename T>
struct Type2Type
{
    typedef T type;
};
template <typename TT>
inline std::complex<TT> conj(std::complex<TT> &tar, Type2Type<std::complex<TT>>)
{
    return std::conj(tar);
}
template <typename TT>
inline TT conj(TT &tar, Type2Type<TT>)
{
    return tar;
}
template <typename T>

class CMatrix
{
private:
    int len_r;                          // Length of row
    int len_c;                          // Length of column
    std::vector<std::vector<T>> matrix; // the Matrix
public:
    CMatrix(int r, int c);                       // Init with 0
    CMatrix(int r, int c, T *matrix_arr);        // Init with array
    CMatrix(int r, int c,  std::vector<std::vector<T>> m); //Init with matrix
    CMatrix(const CMatrix<T> &tar);              // Init with CMatrix
    CMatrix(CMatrix<T> *tar);                    // Init with CMatrix pointer
    void print();                                // Print the matrix
    void operator=(const CMatrix<T> &tar);       // to assign a matrix
    CMatrix<T> operator+(const CMatrix<T> &tar); // matrix add operation
    CMatrix<T> operator-(const CMatrix<T> &tar); // matrix substract operation
    CMatrix<T> operator*(double n);              // matrix scalar multiple
    CMatrix<T> operator*(const CMatrix<T> &tar); // matrix * matrix
    CMatrix<T> operator*(Cvector<T> &tar);       // matrix * vector
    CMatrix<T> operator/(double n);              // matrix scalar divide
    CMatrix<T> el_mult(const CMatrix<T> &tar);   // element-wise matrix multiplication
    T max_el();                                  // find the max element
    T max_el(bool isRow, int idx);               // find the max element in specific axis
    T min_el();                                  // find the min element
    T min_el(bool isRow, int idx);               // find the min element in specific axis
    T avg();                                     // average of all elements
    T avg(bool isRow, int idx);                  // average of specific axis
    T sum();                                     // sum up all elements
    template <typename T2>
    friend T2 det(const CMatrix<T2> &tar); // determinate
    void tranposition();                   // transposition matrix
    void conjugation();                    // conjugation matrix
    ~CMatrix();
	void eigen( CMatrix<T>* eigenVector, T* eigenValue, double precision);
	CMatrix<T> inverse();
	CMatrix<T> reshape(int r, int c);
	CMatrix<T> slice(int r_start, int r_end, int c_start,int c_end);
	CMatrix<T> convolution(CMatrix<T> kernal);
};

template <typename T>
CMatrix<T>::CMatrix(int r, int c)
{
    // 这里需不需要检查正整数？
    try
    {
        if (r <= 0 || c <= 0)
        {
            throw LessThanZeroException();
        }
        len_r = r;
        len_c = c;
        for (int i = 0; i < r; i++)
        {
            std::vector<T> tmp;
            for (int j = 0; j < c; j++)
            {
                tmp.push_back(0);
            }
            matrix.push_back(tmp);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

template <typename T>
CMatrix<T>::CMatrix(int r, int c, T *matrix_arr)
{
    // 这里需不需要检查正整数？
    try
    {
        if (r <= 0 || c <= 0)
        {
            throw LessThanZeroException();
        }

        len_r = r;
        len_c = c;
        for (int i = 0; i < r; i++)
        {
            std::vector<T> tmp;
            for (int j = 0; j < c; j++)
            {
                tmp.push_back(matrix_arr[i * c + j]);
            }
            matrix.push_back(tmp);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}
template <typename T>
CMatrix<T>::CMatrix(const CMatrix<T> &tar)
{
    len_r = tar.len_r;
    len_c = tar.len_c;
    for (int i = 0; i < len_r; i++)
    {
        std::vector<T> tmp;
        for (int j = 0; j < len_c; j++)
        {
            tmp.push_back(tar.matrix[i][j]);
        }
        matrix.push_back(tmp);
    }
}

template <typename T>
CMatrix<T>::CMatrix(CMatrix<T> *tar)
{
    len_r = tar->len_r;
    len_c = tar->len_c;
    for (int i = 0; i < len_r; i++)
    {
        std::vector<T> tmp;
        for (int j = 0; j < len_c; j++)
        {
            tmp.push_back(tar->matrix[i][j]);
        }
        matrix.push_back(tmp);
    }
}
template <typename T>
void CMatrix<T>::print()
{
    std::cout << "--------------start print:---------------" << std::endl;
    for (int i = 0; i < len_r; i++)
    {
        for (int j = 0; j < len_c; j++)
        {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "---------------Done!--------------------" << std::endl;
}

template <typename T>
void CMatrix<T>::operator=(const CMatrix<T> &tar)
{
    len_r = tar.len_r;
    len_c = tar.len_c;
    matrix.clear();
    for (int i = 0; i < len_r; i++)
    {
        std::vector<T> tmp;
        for (int j = 0; j < len_c; j++)
        {
            tmp.push_back(tar.matrix[i][j]);
        }
        matrix.push_back(tmp);
    }
}

template <typename T>
CMatrix<T> CMatrix<T>::operator+(const CMatrix<T> &tar)
{
    try
    {
        if (len_r != tar.len_r || len_c != tar.len_c)
            throw UnalignException();
        CMatrix<T> tmp(len_r, len_c);
        for (int i = 0; i < len_r; i++)
        {
            for (int j = 0; j < len_c; j++)
            {
                tmp.matrix[i][j] = matrix[i][j] + tar.matrix[i][j];
            }
        }
        return tmp;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return CMatrix<T>(1, 1);
    }
}

template <typename T>
CMatrix<T> CMatrix<T>::operator-(const CMatrix<T> &tar)
{
    try
    {
        if (len_r != tar.len_r || len_c != tar.len_c)
            throw UnalignException();
        CMatrix<T> tmp(len_r, len_c);
        for (int i = 0; i < len_r; i++)
        {
            for (int j = 0; j < len_c; j++)
            {
                tmp.matrix[i][j] = matrix[i][j] - tar.matrix[i][j];
            }
        }
        return tmp;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return CMatrix<T>(1, 1);
    }
}
template <typename T>
CMatrix<T> CMatrix<T>::operator*(double n)
{
    CMatrix<T> tmp(len_r, len_c);
    for (int i = 0; i < len_r; i++)
    {
        for (int j = 0; j < len_c; j++)
        {
            tmp.matrix[i][j] = matrix[i][j] * n;
        }
    }
    return tmp;
}
template <typename T>
CMatrix<T> CMatrix<T>::operator*(const CMatrix<T> &tar)
{
    try
    {
        if (len_c != tar.len_r)
            throw UnalignException();
        CMatrix<T> tmp(len_r, len_c);
        for (int i = 0; i < len_r; i++)
        {
            for (int j = 0; j < len_c; j++)
            {
                for (int k = 0; k < len_c; k++)
                {
                    tmp.matrix[i][j] += matrix[i][k] * tar.matrix[k][j];
                }
            }
        }
        return tmp;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return CMatrix<T>(1, 1);
    }
}

template <typename T>
CMatrix<T> CMatrix<T>::operator*(Cvector<T> &tar)
{
    try
    {
        if (len_c != tar.size())
            throw UnalignException();
        CMatrix<T> tmp(len_r, 1);
        for (int i = 0; i < len_r; i++)
        {
            for (int k = 0; k < len_c; k++)
            {
                tmp.matrix[i][0] += matrix[i][k] * tar.getbyIndex(k);
            }
        }
        return tmp;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return CMatrix<T>(1, 1);
    }
}

template <typename T>
CMatrix<T> CMatrix<T>::operator/(double n)
{
    CMatrix<T> tmp(len_r, len_c);
    for (int i = 0; i < len_r; i++)
    {
        for (int j = 0; j < len_c; j++)
        {
            tmp.matrix[i][j] = matrix[i][j] / n;
        }
    }
    return tmp;
}

template <typename T>
CMatrix<T> CMatrix<T>::el_mult(const CMatrix<T> &tar)
{
    try
    {
        if (len_r != tar.len_r || len_c != tar.len_c)
            throw UnalignException();
        CMatrix<T> ansMatrix(this);
        for (int i = 0; i < len_r; i++)
        {
            for (int j = 0; j < len_c; j++)
            {
                ansMatrix.matrix[i][j] = matrix[i][j] * tar.matrix[i][j];
            }
        }
        return ansMatrix;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return CMatrix<T>(1, 1);
    }
}

template <typename T>
T CMatrix<T>::max_el()
{
    T max = matrix[0][0];
    for (int i = 0; i < len_r; i++)
    {
        for (int j = 0; j < len_c; j++)
        {
            if (matrix[i][j] > max)
                max = matrix[i][j];
        }
    }
    return max;
}
template <typename T>
T CMatrix<T>::max_el(bool isRow, int idx)
{
    try
    {
        if (idx < 0 || idx >= len_c * isRow + len_r * (!isRow))
            throw StackOverFlowException();
        T max;
        if (isRow)
        {
            max = matrix[idx][0];
            for (int i = 0; i < len_c; i++)
            {
                if (matrix[idx][i] > max)
                    max = matrix[idx][i];
            }
            return max;
        }
        max = matrix[0][idx];
        for (int i = 0; i < len_r; i++)
        {
            if (matrix[i][idx] > max)
                max = matrix[i][idx];
        }
        return max;
    }
    catch (StackOverFlowException &e)
    {
        std::cerr << e.what() << '\n';
        return (T)(0);
    }
}
template <typename T>
T CMatrix<T>::min_el()
{
    T min = matrix[0][0];
    for (int i = 0; i < len_r; i++)
    {
        for (int j = 0; j < len_c; j++)
        {
            if (matrix[i][j] < min)
                min = matrix[i][j];
        }
    }
    return min;
}
template <typename T>
T CMatrix<T>::min_el(bool isRow, int idx)
{
    try
    {
        if (idx < 0 || idx >= len_c * isRow + len_r * (!isRow))
            throw StackOverFlowException();
        T min;
        if (isRow)
        {
            min = matrix[idx][0];
            for (int i = 0; i < len_c; i++)
            {
                if (matrix[idx][i] < min)
                    min = matrix[idx][i];
            }
            return min;
        }
        min = matrix[0][idx];
        for (int i = 0; i < len_r; i++)
        {
            if (matrix[i][idx] < min)
                min = matrix[i][idx];
        }
        return min;
    }
    catch (StackOverFlowException &e)
    {
        std::cerr << e.what() << '\n';
        return (T)(0);
    }
}
template <typename T>
T CMatrix<T>::avg()
{
    T sum(0);
    for (int i = 0; i < len_r; i++)
    {
        for (int j = 0; j < len_c; j++)
        {
            sum += matrix[i][j];
        }
    }
    return (sum / (len_c * len_r));
}

template <typename T>
T CMatrix<T>::avg(bool isRow, int idx)
{
    try
    {
        if (idx < 0 || idx >= len_c * isRow + len_r * (!isRow))
            throw StackOverFlowException();
        T sum(0);
        if (isRow)
        {
            for (int i = 0; i < len_c; i++)
            {
                sum += matrix[idx][i];
            }
            return sum / len_c;
        }
        for (int i = 0; i < len_c; i++)
        {
            sum += matrix[i][idx];
        }
        return sum / len_r;
    }
    catch (StackOverFlowException &e)
    {
        std::cerr << e.what() << '\n';
        return (T)(0);
    }
}
template <typename T>
T CMatrix<T>::sum()
{
    T sum(0);
    for (int i = 0; i < len_r; i++)
    {
        for (int j = 0; j < len_c; j++)
        {
            sum += matrix[i][j];
        }
    }
    return sum;
}

template <typename T>
void CMatrix<T>::tranposition()
{
    CMatrix<T> rawMatrix(this);
    std::swap(len_r, len_c);
    this->matrix.clear();
    for (int i = 0; i < len_r; i++)
    {
        std::vector<T> tmp;
        for (int j = 0; j < len_c; j++)
        {
            tmp.push_back(rawMatrix.matrix[j][i]);
        }
        matrix.push_back(tmp);
    }
}
template <typename T>
T det(const CMatrix<T> &tar)
{
    try
    {
        if (tar.len_c != tar.len_r)
            throw UnalignException();
        if (tar.len_r == 1)
            return tar.matrix[0][0];
        T d = T(0);
        for (int i = 0; i < tar.len_c; i++)
        {
            CMatrix<T> tmp = CMatrix<T>(tar.len_r - 1, tar.len_r - 1);
            tmp.matrix.clear();
            for (int j = 1; j < tar.len_r; j++)
            {
                std::vector<T> t;
                for (int k = 0; k < tar.len_c; k++)
                {
                    if (k != i)
                    {
                        t.push_back(tar.matrix[j][k]);
                    }
                }
                tmp.matrix.push_back(t);
            }
            d += pow(-1,i)*tar.matrix[0][i] * det(tmp);
        }
        return d;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
}
template <typename T>
void CMatrix<T>::conjugation()
{

    CMatrix<T> rawMatrix(this);
    this->matrix.clear();
    for (int i = 0; i < len_r; i++)
    {
        std::vector<T> tmp;
        for (int j = 0; j < len_c; j++)
        {
            tmp.push_back(conj(rawMatrix.matrix[i][j], Type2Type<T>()));
        }
        matrix.push_back(tmp);
    }
}
//jacobi eigenvalue algorithm    该算法只能算对称矩阵！！！
template <typename T>
void CMatrix<T>::eigen( CMatrix<T>* eigenVector, T* eigenValue, double precision) {
	try {
		if(this->len_c != this->len_r) throw InvalidException();

		CMatrix<T> ma(*this);// copy matrix
		int n = ma.len_c;
		std::vector<std::vector<T>> *eigenV = &(eigenVector->matrix);
		for (int i = 0; i < n; i++){
			for(int j = 0; j < n;j++){
				if(i==j) (*eigenV)[i][j] = (T)1;
				else (*eigenV)[i][j] = (T)0;
			}
		}

		for(int m = 0; m < 1000000 ; m++) {
			T max = ma.matrix[0][1];
			int maxR = 0;
			int maxC = 1;
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i != j && fabs(ma.matrix[i][j]) > max) {
						maxR = i;
						maxC = j;
						max = fabs(ma.matrix[i][j]);
					}
				}
			}

			if (max < precision) break;

			T angle =
					0.5 * atan2(-2 * ma.matrix[maxR][maxC], ma.matrix[maxC][maxC] - ma.matrix[maxR][maxR]);
			T sinT = sin(angle);
			T cosT = cos(angle);
			T sinT2 = sin(2 * angle);
			T cosT2 = cos(2 * angle);
			T Sii = ma.matrix[maxR][maxR];
			T Sij = ma.matrix[maxR][maxC];
			T Sjj = ma.matrix[maxC][maxC];
			ma.matrix[maxR][maxR] = Sii * cosT * cosT + 2 * Sij * sinT * cosT + Sjj * sinT * sinT;
			ma.matrix[maxC][maxC] = Sii * sinT * sinT - 2 * Sij * sinT * cosT + Sjj * cosT * cosT;
			ma.matrix[maxR][maxC] = ma.matrix[maxC][maxR] = cosT2 * Sij + 0.5 * sinT2 * (Sjj - Sii);

			for (int i = 0; i < n; i++) {
				if (i != maxC && i != maxR) {
					T temp1 = ma.matrix[i][maxC];
					T temp2 = ma.matrix[i][maxR];
					ma.matrix[i][maxR] = temp1*sinT + temp2*cosT;
					ma.matrix[i][maxC] = temp1*cosT - temp2*sinT;

					T temp3 = ma.matrix[maxC][i];
					T temp4 = ma.matrix[maxR][i];
					ma.matrix[maxR][i] = temp3*sinT + temp4*cosT;
					ma.matrix[maxC][i] = temp3*cosT - temp4*sinT;
				}
			}

			//eigen vector
			for(int i = 0; i < n;i++){
				T temp = (*eigenV)[i][maxR];
				(*eigenV)[i][maxR] = (*eigenV)[i][maxC]*sinT + temp * cosT;
				(*eigenV)[i][maxC] = (*eigenV)[i][maxC]*cosT - temp*sinT;
			}
			//eigen value
			for(int i = 0; i < n;i++){
				T ans = 0;
				for(int j = 0; j < n;j++){
					ans+=this->matrix[0][j]*(*eigenV)[j][i];
				}
				eigenValue[i] = ans/(*eigenV)[0][i];
			}
		}
	}catch(InvalidException & e){
		std::cerr<<e.what()<<std::endl;
	}
}

template <typename T>
CMatrix<T>::~CMatrix()
{
}
//use adjoint matrix to get inverse
template<typename T>
CMatrix<T> CMatrix<T>::inverse() {
	try {
		if(len_r!=len_c) throw InvalidException();
		int n = len_c;
		//initialize adjoint matrix
		T arr[n*n];
		for(int i = 0; i < n*n;i++){
			arr[i]=0;
		}
		for(int i = 0; i < n;i++){
			arr[i*n+i]=1;
		}

		this->print();
		std::vector<std::vector<T>> copy = matrix;
		for(int i = 0; i < n;i++){
			for(int j = i+1; j < n;j++){
				double t =  copy[j][i]/copy[i][i] ;
				for(int k = 0; k<n ; k++){
					copy[j][k] = copy[j][k] - t * copy[i][k];
					arr[j*n+k] = arr[j*n+k] - t * arr[i*n+k];
				}
			}
		}
		for(int i = n-1;i>=0;i--){
			for(int j = i-1 ; j >= 0 ;j--){
				double t = copy[j][i]/copy[i][i];
				for(int k = n-1;k>=0;k--){
					copy[j][k] = copy[j][k] - t * copy[i][k];
					arr[j*n+k] = arr[j*n+k] - t * arr[i*n+k];
				}
			}
		}
		for(int i = 0; i < n;i++){
			double t = copy[i][i];
			for(int j = 0; j < n;j++){
				arr[i*n+j]/=t;
				copy[i][j]/=t;
			}
		}
		CMatrix<T> inverseMatrix(n,n,arr);
		return inverseMatrix;
	}catch(InvalidException & e){
		std::cerr<<e.what()<<std::endl;
	}
}

template<typename T>
CMatrix<T>::CMatrix(int r, int c,  std::vector<std::vector<T>> m){
	try {
		if(r<=0||c<=0) throw LessThanZeroException();
		len_r = r;
		len_c = c;
		matrix = m;
	}catch(LessThanZeroException & e){
		std::cerr<<e.what()<<std::endl;
	}
}

template<typename T>
CMatrix<T> CMatrix<T>::reshape(int r, int c) {
	try {
		if(r*c!=len_r*len_c) throw InvalidException();

		int l = len_r*len_c;
		T arr[l];
		for(int i = 0; i < len_r;i++){
			for(int j = 0; j < len_c;j++){
				arr[i*len_c+j] = matrix[i][j];
			}
		}
		return CMatrix<T>(r,c,arr);
	}catch(InvalidException & e){
		std::cerr<<e.what()<<std::endl;
	}
}

template<typename T>
CMatrix<T> CMatrix<T>::slice(int r_start, int r_end, int c_start,int c_end) {
	try {
		if(r_end>=len_r || c_end >= len_c || r_start<0 || c_start <0) throw InvalidException();

		int r = r_end - r_start + 1;
		int c = c_end - c_start + 1;
		T arr[r*c];
		int m = 0;
		for(int i = r_start; i <= r_end;i++){
			for(int j = c_start; j <= c_end;j++){
				arr[m] = matrix[i][j];
				m++;
			}
		}
		for(int i = 0; i < r*c;i++)std::cout<< arr[i] <<" ";
		std::cout<<std::endl;
		return CMatrix<T>(r,c,arr);
	}catch(InvalidException & e){
		std::cerr<<e.what()<<std::endl;
	}
}

template<typename T>
CMatrix<T> CMatrix<T>::convolution(CMatrix<T> kernal) {
	try{
		int rkernal = kernal.len_r;
		int ckernal = kernal.len_c;
		int rmat = this->len_r+2*(rkernal-1);
		int cmat = this->len_c+2*(rkernal-1);
		int rans = rmat-rkernal+1;
		int cans = cmat-ckernal+1;
		T ans[rans*cans];
		std::vector<std::vector<T>> m; // pad 0 to the original mat
		for (int i = 0; i < rmat; i++)
		{
			std::vector<T> tmp;
			for (int j = 0; j < cmat; j++)
			{
				tmp.push_back(0);
			}
			m.push_back(tmp);
		}
		for(int i = 0;i < this->len_r;i++){
			for (int j = 0; j < this->len_c; j++)
			{
				m[i+rkernal-1][j+ckernal-1] = this->matrix[i][j];
			}
		}

		T t;
		for(int i = 0; i < rans;i++){
			for(int j = 0; j < cans;j++){
				t = 0;
				for(int k = 0; k < rkernal;k++){
					for(int n = 0; n < ckernal;n++){
						t += m[k+i][n+j] * kernal.matrix[rkernal-1-k][ckernal-1-n]; // to save to rotation
					}
				}
				ans[i*cans+j] = t;
			}
		}

		return CMatrix<T>(rans,cans,ans);

	}catch(InvalidException & e){
		std::cerr<<e.what()<<std::endl;
	}

}

#endif
