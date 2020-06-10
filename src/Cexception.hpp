#ifndef CEXCEPTION_HPP
#define CEXCEPTION_HPP
#include <iostream>
#include <exception>

class UnalignException
{
public:
    const char *what() const throw()
    {
        return "UnalignException: The vectors or Matrices are not aligned.";
    }
};

class LessThanZeroException
{
public:
    const char *what() const throw()
    {
        return "LessThanZeroException: The size is less than 0.";
    }
};
class InvalidException
{
public:
    const char *what() const throw()
    {
        return "InvalidException: the arguments are invalid.";
    }
};

class StackOverFlowException
{
public:
    const char *what() const throw()
    {
        return "StackOverFlow: the index is out of boundary.";
    }
};
#endif