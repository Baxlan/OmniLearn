#ifndef BRAIN_EXCEPTION_HH_
#define BRAIN_EXCEPTION_HH_

#include <string>
#include <exception>

namespace brain
{



struct Exception : public std::exception
{
    Exception(std::string const& msg):
    _msg("[Brain.Exception : " + msg + "]")
    {
    }

    virtual ~Exception()
    {
    }

    virtual const char* what() const noexcept
    {
        return _msg.c_str();
    }

private:
    std::string const _msg;
};



} // namespace brain

#endif // BRAIN_EXCEPTION_HH_