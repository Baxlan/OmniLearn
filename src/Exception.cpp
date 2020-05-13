// Exception.cpp

#include "omnilearn/Exception.hh"



omnilearn::Exception::Exception(std::string const& msg):
_msg(msg)
{
}


const char* omnilearn::Exception::what() const noexcept
{
    return std::string("[OmniLearn.Exception : " + _msg + "]").c_str();
}