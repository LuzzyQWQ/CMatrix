#ifndef CVECTOR_HPP
#define CVECTOR_HPP
#include <iostream>
#include <cstdlib>
#include <vector>
#include <math.h>
#include "Cexception.hpp"

template <typename T>
class Cvector;

template <typename T>
class Cvector
{
private:
    std::vector<T> vec;

public:
    Cvector();                     // Init a 1 size vector as (1,1...,1)
    Cvector(int n);                     // Init a n size vector as (1,1...,1)
    Cvector(const std::vector<T> &tar); // Init with a vector
    Cvector(const Cvector<T> &tar);     // Init with a Cvector
    void print();                       // print the vector

    // Operations
    void operator=(const Cvector<T> &tar);       // to assign a vector
    Cvector<T> operator+(const Cvector<T> &tar); // vector add
    Cvector<T> operator-(const Cvector<T> &tar); // vector substract
    Cvector<T> operator*(const double tar);      // scalar multiple "vec * 3"
    Cvector<T> operator/(const double tar);      // divide a scalar "vec / 3"
    int size();                                  // size of vector
    T getbyIndex(int idx);                       // get element by index;
    T module();                                  // The module of the vector
    T dot(const Cvector<T> &tar);                // dot multiple
    Cvector<T> cross(const Cvector<T> &tar);     // cross multiple
    Cvector<T> el_mult(const Cvector<T> &tar);   // element-wise multiplication
    template <typename T2>
    friend Cvector<T2> operator*(double n, const Cvector<T2> &tar); // scalar multiple "3 * vec"
    ~Cvector();
};
template <typename T>
Cvector<T>::Cvector()
{   
    vec.push_back(1);
}
template <typename T>
Cvector<T>::Cvector(int n)
{
    for (int i = 0; i < n; i++)
    {
        vec.push_back(1);
    }
}
template <typename T>
Cvector<T>::Cvector(const std::vector<T> &tar)
{
    vec.assign(tar.begin(), tar.end());
}

template <typename T>
Cvector<T>::Cvector(const Cvector<T> &tar)
{
    vec.assign(tar.vec.begin(), tar.vec.end());
}

template <typename T>
void Cvector<T>::print()
{
    std::cout << "(";
    if (vec.empty())
    {
        std::cout << ")" << std::endl;
        return;
    }
    for (int i = 0; i < vec.size() - 1; i++)
    {
        std::cout << vec[i] << ", ";
    }
    std::cout << vec[vec.size() - 1];
    std::cout << ")" << std::endl;
}

// Operations
template <typename T>
void Cvector<T>::operator=(const Cvector<T> &tar)
{
    vec.clear();
    for (int i = 0; i < tar.vec.size(); i++)
    {
        vec.push_back(tar.vec[i]);
    }
}

template <typename T>
Cvector<T> Cvector<T>::operator+(const Cvector<T> &tar)
{
    try
    {
        // check whether align
        if (vec.size() != tar.vec.size())
        {
            throw UnalignException();
        }
        Cvector<T> tmp(vec.size());
        for (int i = 0; i < vec.size(); i++)
        {
            tmp.vec[i] = vec[i] + tar.vec[i];
        }
        return tmp;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return Cvector<T>(0);
    }
}

template <typename T>
Cvector<T> Cvector<T>::operator-(const Cvector<T> &tar)
{
    try
    {
        // check whether align
        if (vec.size() != tar.vec.size())
        {
            throw UnalignException();
        }
        Cvector<T> tmp(vec.size());
        for (int i = 0; i < vec.size(); i++)
        {
            tmp.vec[i] = vec[i] - tar.vec[i];
        }
        return tmp;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return Cvector<T>(0);
    }
}

template <typename T>
Cvector<T> Cvector<T>::operator*(const double tar)
{
    Cvector<T> tmp(vec.size());
    for (int i = 0; i < vec.size(); i++)
    {
        tmp.vec[i] = vec[i] * tar;
    }
    return tmp;
}

template <typename T>
Cvector<T> Cvector<T>::operator/(const double tar)
{
    Cvector<T> tmp(vec.size());
    for (int i = 0; i < vec.size(); i++)
    {
        tmp.vec[i] = vec[i] / tar;
    }
    return tmp;
}
template <typename T>
int Cvector<T>::size()
{
    return vec.size();
}
template <typename T>
T Cvector<T>::getbyIndex(int idx)
{
    try
    {
        if (idx<0||idx>=vec.size())
            throw StackOverFlowException();
        return vec[idx];
    }
    catch (StackOverFlowException &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
}
template <typename T>
T Cvector<T>::module()
{
    T ans(0);
    for (int i = 0; i < vec.size(); i++)
    {
        ans += vec[i] * vec[i];
    }
    return sqrt(ans);
}

template <typename T>
T Cvector<T>::dot(const Cvector<T> &tar)
{
    try
    {
        // check whether align
        if (vec.size() != tar.vec.size())
        {
            throw UnalignException();
        }
        T ans(0);
        for (int i = 0; i < vec.size(); i++)
        {
            ans += vec[i] * tar.vec[i];
        }
        return ans;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return T(-1);
    }
}
template <typename T>
Cvector<T> Cvector<T>::cross(const Cvector<T> &tar)
{
    try
    {
        // check whether align
        if (vec.size() != tar.vec.size())
        {
            throw UnalignException();
        }
        if (vec.size() != 3)
        {
            throw InvalidException();
        }
        // TODO
        Cvector<T> tmp(3);
        tmp.vec[0] = vec[1] * tar.vec[2] - vec[2] * tar.vec[1];
        tmp.vec[1] = vec[2] * tar.vec[0] - vec[0] * tar.vec[2];
        tmp.vec[2] = vec[0] * tar.vec[1] - vec[1] * tar.vec[0];
        return tmp;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return Cvector<T>(0);
    }
    catch (InvalidException &e)
    {
        std::cerr << e.what() << '\n';
        return Cvector<T>(0);
    }
}

template <typename T>
Cvector<T> Cvector<T>::el_mult(const Cvector<T> &tar)
{
    try
    {
        // check whether align
        if (vec.size() != tar.vec.size())
        {
            throw UnalignException();
        }
        Cvector<T> ans(vec.size());
        for (int i = 0; i < vec.size(); i++)
        {
            ans.vec[i] = vec[i] * tar.vec[i];
        }
        return ans;
    }
    catch (UnalignException &e)
    {
        std::cerr << e.what() << '\n';
        return Cvector<T>(0);
    }
}
template <typename T>
Cvector<T> operator*(double n, const Cvector<T> &tar)
{
    Cvector<T> tmp(tar.vec.size());
    for (int i = 0; i < tar.vec.size(); i++)
    {
        tmp.vec[i] = tar.vec[i] * n;
    }
    return tmp;
}

template <typename T>
Cvector<T>::~Cvector()
{
}

#endif