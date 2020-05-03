// Exception.hh

#ifndef OMNILEARN_EXCEPTION_HH_
#define OMNILEARN_EXCEPTION_HH_

#include <exception>
#include <fstream>
#include <string>



namespace omnilearn
{



struct Exception : public std::exception
{
    Exception(std::string const& msg);
    virtual const char* what() const noexcept;

protected:
    std::string const _msg;
};



struct LogException : public Exception
{
    LogException(std::string const& msg);
    void log(std::ofstream& file) const;
};



} // namespace omnilearn



#endif // OMNILEARN_EXCEPTION_HH_