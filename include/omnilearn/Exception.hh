// Exception.hh

#ifndef OMNILEARN_EXCEPTION_HH_
#define OMNILEARN_EXCEPTION_HH_

#include <exception>
#include <string>



namespace omnilearn
{



struct Exception : public std::exception
{
    Exception(std::string const& msg);
    virtual const char* what() const noexcept;

private:
    std::string const _msg;
};



} // namespace omnilearn

#endif // OMNILEARN_EXCEPTION_HH_